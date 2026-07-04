# ULSIR
This repository contains a pytorch implementation for the paper: ULSIR: Dual-Frequency Disentanglement for Ultra-Low-Light Satellite Image Restoration

## Dataset and Pre-trained Models
Please download iSAID Dataset ([IASID Dataset](https://drive.google.com/file/d/1mlTTdbqG1ZheaWsBcIjAKDyCdbuAqpvy/view)), then place them in the project trainsets directory. 

Please download LLSD Dataset ([LLSD Dataset](https://pan.baidu.com/s/15jmuwFR5wboHXnmMn9tYBA))(code:c7bh), then place them in the project trainsets directory. 

Please download pre-trained models ([premodel](https://pan.baidu.com/s/1eXYkUluDOUhu-mGMp6ZfMA))(code: k7kx), and then place the `.pth` in the project premodel directory.

## Environment
```bash
conda create -n ulsir python=3.10  # (Python >= 3.8)
conda activate ulsir
pip install -r requirements.txt
```
## Train
```bash
python train.py --option options/ulsir.ymal
```
## Inference
```bash
python test.py  --premodel *.pth
```
