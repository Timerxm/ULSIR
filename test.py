import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
from models.darkcopy import LowLightEnhancer as net
#from models.dark import LowLightEnhancer as net
#from models.retinexFormer import RetinexFormer as net
#from models.ddfn import DDFN as net
#from models.darkorg import DarkIR as net  
from huawei_utils import util_calculate_psnr_ssim as util
from huawei_utils.utils_modelsummary import  *
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
torch.cuda.set_device(0)  # The selected CUDA_VISIBLE_DEVICES entry maps to device index 0.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,default='superresolution/fftwave/models/1441_G.pth' \
    '')
    parser.add_argument('--save_dir', type=str,default='result/raunyuzhit/')
    parser.add_argument('--folder_lq', type=str, default='/media/user/data/mcg/data/RS/LLRS/val/low2', help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default='/media/user/data/mcg/data/RS/LLRS/val/low2', help='input ground-truth test image folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = net(img_channel=1,
                 width=32, 
                 middle_blk_num_enc=2,
                 middle_blk_num_dec=2, 
                 enc_blk_nums=[1, 2, 3], 
                 dec_blk_nums=[3, 2, 1],  
                 dilations = [1, 4, 9], 
                 extra_depth_wise = True)

    param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        imgname, imgext, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32

        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        with torch.no_grad():
            output = model(img_lq)[0]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        
        save_path = f'{save_dir}/{imgname}{imgext}'
        cv2.imwrite(save_path, output)
        print(f"已保存增强图片至: {save_path}")

def setup(args):
    save_dir = f'{args.save_dir}'
    folder = args.folder_gt
    return folder, save_dir

def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    img_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    if img_gt.ndim == 2:
            img_gt = np.expand_dims(img_gt, axis=-1)  # (H,W) → (H,W,1)
    elif img_gt.ndim == 3 and img_gt.shape[2] == 3:
            pass 
    
    lq_img_path = os.path.join(args.folder_lq, f"{imgname}{imgext}")
    img_lq = cv2.imread(lq_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    if img_lq.ndim == 2:
            img_lq = np.expand_dims(img_lq, axis=-1)
    elif img_lq.ndim == 3 and img_lq.shape[2] == 3:
            pass

    return imgname, imgext, img_lq, img_gt

if __name__ == '__main__':
    main()
