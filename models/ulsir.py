import torch
import torch.nn as nn
import sys
import os
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
root_dir = os.path.dirname(current_dir) 
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch.nn.functional as F
import math
import time

class CustomSequential(nn.Module):
    '''
    Similar to nn.Sequential, but it lets us introduce a second argument in the forward method 
    so adaptors can be considered in the inference.
    '''
    def __init__(self, *args):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x,  *args, **kwargs):
        for module in self.modules_list:
            x = module(x, *args, **kwargs)
        return x

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''

    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c

        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation,
                      stride=1, groups=self.dw_channel,
                      bias=True, dilation=dilation),  # the dconv
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, input):
        return self.branch(input)

class EBlock(nn.Module):
    '''
    Change this block using Branch
    '''

    def __init__(self, c, DW_Expand=2, dilations=[1], extra_depth_wise=False):
        super().__init__()
        # we define the 2 branches
        self.dw_channel = DW_Expand * c
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True,dilation=1) if extra_depth_wise else nn.Identity()  # optional extra dw
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,groups=1, bias=True, dilation=1)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation=dilation))

        self.sg1 = SimpleGate()
        self.wave = WaveletAttention(channels=self.dw_channel // 2 , use_fc=True)

        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2 , out_channels=c, kernel_size=1, padding=0, stride=1,groups=1, bias=True, dilation=1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=2 * c, kernel_size=1, padding=0, stride=1, groups=1,bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        
    def forward(self, inp):
        y = inp
        x = self.norm1(inp)

        x = self.conv1(self.extra_conv(x))

        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.wave(z)
        x = self.conv3(x)
        y = inp + self.beta * x

        # second step
        x_step2 = self.norm2(y)  # size [B, 2*C, H, W]
        x = self.conv4(x_step2)
        x = self.sg2(x)
        x = self.conv5(x)
        x = y + x * self.gamma

        return x
    
def build_wavelet_kernels(device=None, dtype=torch.float32):                                                                                                                                                                       

    s = 1.0 / math.sqrt(2.0)
    h0 = torch.tensor([s, s], dtype=dtype, device=device)      # Low-pass                                                                                                                                                                       
    h1 = torch.tensor([-s, s], dtype=dtype, device=device)     # High-pass                                                                                                                                                                       
    # Use outer products to obtain 2x2 kernels.
    LL = torch.ger(h0, h0)  # Low-low
    LH = torch.ger(h0, h1)  # Low-high (vertical edges)
    HL = torch.ger(h1, h0)  # High-low (horizontal edges)
    HH = torch.ger(h1, h1)  # High-high (diagonal details)
    # Keep the shape as (1,1,2,2) for later expansion to groups=C.
    filt = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)                                                                                                                                                                       
    return filt  # (4,1,2,2)


# =========================
# Wavelet Attention module
# =========================
class WaveletAttention(nn.Module):
    """
    Processing steps:
    X --DWT--> (LH, HL, HH, LL)
         High-frequency thresholding -> concat -> 1x1 conv fusion -> IDWT with LL -> X_re
         GAP -> optional FC -> Softmax -> channel weights
         Output: Final = weight * X
    """
    def __init__(self, channels, use_fc=True):
        super().__init__()
        self.channels = channels
        self.use_fc = use_fc

        # Soft-threshold parameters (3 high-frequency subbands * C), constrained to 0-1 by sigmoid and scaled by mean(|x|).
        self.theta = nn.Parameter(torch.zeros(3, channels, 1, 1))

        # High-frequency subband fusion: 3C -> C.
        self.fuse = nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False)                                                                                                                                                                       

        # Optional FC after GAP, preserving C -> C.
        if use_fc:
            self.fc = nn.Linear(channels, channels, bias=True)

        # Wavelet kernels registered as buffers, moved by to(device) but not trained.
        filt = build_wavelet_kernels()
        self.register_buffer("w_analysis", filt)   # (4,1,2,2)
        self.register_buffer("w_synthesis", filt)  # Db1 is orthogonal: synthesis equals analysis.

    # ---------- DWT and IDWT ----------
    def dwt(self, x):
        """
        x: (B,C,H,W)
        Return LH, HL, HH, LL and intermediate size information.
        """
        B, C, H, W = x.shape

        # Zero-pad to even spatial dimensions to avoid boundary loss.
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)                                                                                                                                                                       

        # Group convolution: each channel uses the same set of 4 filters.
        # Expand the weight shape to (4*C, 1, 2, 2) with groups=C.
        weight = self.w_analysis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        y = F.conv2d(x, weight=weight, bias=None, stride=2, padding=0, groups=C)  # (B,4C,H/2,W/2)                                                                                                                                                                       

        # Split by subband.
        y = y.view(B, C, 4, y.size(-2), y.size(-1)).contiguous()                                                                                                                                                                       
        LL = y[:, :, 0]  # (B,C,h,w)
        LH = y[:, :, 1]
        HL = y[:, :, 2]
        HH = y[:, :, 3]
        return LH, HL, HH, LL

    def idwt(self, LH, HL, HH, LL):
        """
        Inverse transform: reconstruct the four subbands as (B,C,H,W).
        """
        B, C, h, w = LL.shape
        # Stack the 4 subbands back to (B,4C,h,w).
        y = torch.stack([LL, LH, HL, HH], dim=2).view(B, 4 * C, h, w)

        # Use conv_transpose2d as the synthesis filter with stride=2.
        weight = self.w_synthesis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        # conv_transpose weight shape: (in_channels, out_channels/groups, kH, kW).
        # We use groups=C so each group synthesizes 4 subbands into 1 channel.
        # Treat weight as (4C, 1, 2, 2); groups=C maps every 4 inputs to 1 output.
        x_rec = F.conv_transpose2d(y, weight=weight, bias=None, stride=2, padding=0, groups=C)                                                                                                                                                                       
        return x_rec

    # ---------- High-frequency soft threshold ----------
    @staticmethod
    def soft_threshold(x, thr):
        # soft-shrinkage： sign(x) * relu(|x| - thr)
        return torch.sign(x) * F.relu(torch.abs(x) - thr)

    # ---------- Forward ----------
    def forward(self, x):
        B, C, H, W = x.shape

        # 1) DWT
        LH, HL, HH, LL = self.dwt(x)

        # 2) Threshold and fuse high-frequency subbands. The normalized per-channel threshold is constrained to 0-1 and scaled by the subband mean magnitude.
        eps = 1e-6
        m_LH = LH.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HL = HL.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HH = HH.abs().mean(dim=(2, 3), keepdim=True) + eps

        t = torch.sigmoid(self.theta)  # (3,C,1,1)
        thr_LH = t[0].unsqueeze(0) * m_LH
        thr_HL = t[1].unsqueeze(0) * m_HL
        thr_HH = t[2].unsqueeze(0) * m_HH

        LH_hat = self.soft_threshold(LH, thr_LH)
        HL_hat = self.soft_threshold(HL, thr_HL)
        HH_hat = self.soft_threshold(HH, thr_HH)

        # Fusion convolution (3C -> C).
        H_concat = torch.cat([LH_hat, HL_hat, HH_hat], dim=1)  # (B,3C,h,w)                                                                                                                                                                       
        H_fused = self.fuse(H_concat)  # (B,C,h,w)

        # 3) IDWT reconstruction.
        X_re = self.idwt(LH_hat, HL_hat, H_fused, LL)  # (B,C,H',W')，H'/W'≈H/W                                                                                                                                                                       
        
        # 4) Attention weights: GAP -> optional FC -> Softmax along channels.
        gap = F.adaptive_avg_pool2d(X_re, 1).view(B, C)  # (B,C)
        if self.use_fc:
            gap = self.fc(gap)  # (B,C)
        attn = F.softmax(gap, dim=1).view(B, C, 1, 1)  # (B,C,1,1)
        
        # 5) Weight the original input.
        out = x * attn
        
        return out

class OurFFN(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(OurFFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim * 2
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp),
        )

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
        )


    def forward(self, x):
        x = self.conv_init(x)
        x = self.conv1_1(x)
        x = self.gelu(x)
        x = self.conv_fina(x)

        return x

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.dim = in_channels

        self.conv_layer = nn.Sequential(
                                        nn.BatchNorm2d(out_channels * 2),
                                        nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                                        kernel_size=1, stride=1, padding=0, groups=self.groups,bias=True),
                                        nn.GELU(),
                                        )


    def forward(self, x):
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output


class OurTokenMixer_For_Gloal(nn.Module):
    def __init__(
            self,
            dim
    ):
        super(OurTokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1)
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.FFC(x)
        x = self.conv_fina(x)

        return x


class OurMixer(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_gloal=OurTokenMixer_For_Gloal
    ):
        super(OurMixer, self).__init__()
        self.dim = dim
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
        )

        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU()
        )


    def forward(self, x):
        x = self.conv_init(x)
        x = self.mixer_gloal(x)
        x = self.gelu(x)
        x = self.ca_conv(x)

        return x


class OurBlock(nn.Module):
    def __init__(
            self,
            dim,
            norm_layer=nn.BatchNorm2d,
            token_mixer=OurMixer
    ):
        super(OurBlock, self).__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mixer = token_mixer(dim=self.dim)
        self.ffn = OurFFN(dim=self.dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x * self.beta + copy

        copy = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x * self.gamma + copy

        return x


# need drop_path?
class OurStage(nn.Module):
    def __init__(
            self,
            depth=int,
            in_channels=int,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(OurStage, self).__init__()
        # Init blocks
        self.blocks = nn.Sequential(*[
                OurBlock(
                    dim=in_channels,
                    norm_layer=nn.BatchNorm2d,
                    token_mixer=OurMixer
                )
            for index in range(depth)
        ])

    def forward(self, input=torch.Tensor) -> torch.Tensor:
        output = self.blocks(input)
        return output
    

class Converse2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale=1, padding=2, padding_mode='circular', eps=1e-5):                                                                                                                                                                      
        super(Converse2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size =  kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.eps = eps


        # ensure depthwise
        assert self.out_channels == self.in_channels
        self.weight = nn.Parameter(torch.randn(1, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        self.weight.data = nn.functional.softmax(self.weight.data.view(1,self.in_channels,-1), dim=-1).view(1, self.in_channels, self.kernel_size, self.kernel_size)                                                                                                                                                                       

        
    def forward(self, x):

        if self.padding > 0:
            x = nn.functional.pad(x, pad=[self.padding, self.padding, self.padding, self.padding], mode=self.padding_mode, value=0)                                                                                                                                                                       

        self.biaseps = torch.sigmoid(self.bias-9.0) + self.eps
        _, _, h, w = x.shape
        STy = self.upsample(x, scale=self.scale)
        if self.scale != 1:
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')                                                                                                                                                                       

        FB = self.p2o(self.weight, (h*self.scale, w*self.scale))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        FBFy = FBC*torch.fft.fftn(STy, dim=(-2, -1))
        
        FR = FBFy + torch.fft.fftn(self.biaseps*x, dim=(-2,-1))
        x1 = FB.mul(FR)
        FBR = torch.mean(self.splits(x1, self.scale), dim=-1, keepdim=False)                                                                                                                                                                       
        invW = torch.mean(self.splits(F2B, self.scale), dim=-1, keepdim=False)                                                                                                                                                                       
        invWBR = FBR.div(invW + self.biaseps)
        FCBinvWBR = FBC*invWBR.repeat(1, 1, self.scale, self.scale)
        FX = (FR-FCBinvWBR)/self.biaseps
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        if self.padding > 0:
            out = out[..., self.padding*self.scale:-self.padding*self.scale, self.padding*self.scale:-self.padding*self.scale]                                                                                                                                                                       

        return out

    def splits(self, a, scale):
        *leading_dims, W, H = a.size()
        W_s, H_s = W // scale, H // scale

        # Reshape to separate the scale factors
        b = a.view(*leading_dims, scale, W_s, scale, H_s)

        # Generate the permutation order
        permute_order = list(range(len(leading_dims))) + [len(leading_dims) + 1, len(leading_dims) + 3, len(leading_dims), len(leading_dims) + 2]                                                                                                                                                                       
        b = b.permute(*permute_order).contiguous()

        # Combine the scale dimensions
        b = b.view(*leading_dims, W_s, H_s, scale * scale)
        return b


    def p2o(self, psf, shape):
        otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
        otf[...,:psf.shape[-2],:psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(psf.shape[-2]/2), -int(psf.shape[-1]/2)), dims=(-2, -1))                                                                                                                                                                       
        otf = torch.fft.fftn(otf, dim=(-2,-1))

        return otf

    def upsample(self, x, scale=3):
        st = 0
        z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*scale, x.shape[3]*scale)).type_as(x)                                                                                                                                                                       
        z[..., st::scale, st::scale].copy_(x)
        return z


class LowLightEnhancer(nn.Module):

    def __init__(self, img_channel=3,
                 width=32,
                 middle_blk_num_enc=2,
                 middle_blk_num_dec=2,
                 enc_blk_nums=[1, 2, 3],
                 dec_blk_nums=[3, 1, 1],
                 dilations=[1, 4, 9],
                 extra_depth_wise=True):
        super(LowLightEnhancer, self).__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,groups=1,bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width  # dim
        for num in enc_blk_nums:
            self.encoders.append(
                CustomSequential(
                                *[module for _ in range(num) for module in (
                                    OurStage(depth=1, in_channels=chan),
                                    EBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise),
                                    )]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks_enc = \
            CustomSequential(
                            *[module for _ in range(middle_blk_num_enc) for module in (
                                    OurStage(depth=1, in_channels=chan),
                                    EBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise),
                                    )]
            )
        self.middle_blks_dec = \
            CustomSequential(
                            *[module for _ in range(middle_blk_num_dec) for module in (
                                    OurStage(depth=1, in_channels=chan),
                                    EBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise),
                                    )]
            )

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                Converse2D(in_channels=chan, out_channels=chan, kernel_size=5, scale=2),
                nn.Conv2d(chan, chan // 2, 1, bias=False),
                #nn.Conv2d(chan, chan * 2, 1, bias=False),
                #nn.PixelShuffle(2)
            )
            )           
            chan = chan // 2
            self.decoders.append(
                CustomSequential(
                                *[module for _ in range(num) for module in (
                                    OurStage(depth=1, in_channels=chan),
                                    EBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise),
                                    )]
                )
            )
        self.padder_size = 2 ** len(self.encoders)

        self.side_out = nn.Conv2d(in_channels=width * 2 ** len(self.encoders), out_channels=img_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        side_loss=True
        _, _, H, W = input.shape

        input = self.check_image_size(input)
        
        x = self.intro(input)

        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        # we apply the encoder transforms
        x_light = self.middle_blks_enc(x)

        if side_loss:
            out_side = self.side_out(x_light)
        # apply the decoder transforms
        x = self.middle_blks_dec(x_light)
        x = x + x_light

        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)

        x = x + input

        out = x[:, :, :H , :W ]  # we recover the original size of the image

        if side_loss:
            return out , out_side
        else:
            return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value = 0)
        return x

