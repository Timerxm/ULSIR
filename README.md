# ULSIR
This repository contains a pytorch implementation for the paper: ULSIR: Dual-Frequency Disentanglement for Ultra-Low-Light Satellite Image Restoration

## Dataset and Pre-trained Models
Please download Training Dataset ([IASID Dataset](https://drive.google.com/file/d/1mlTTdbqG1ZheaWsBcIjAKDyCdbuAqpvy/view)) and Testing Dataset ([Baidu Disk](https://pan.baidu.com/s/1QbvWxKWccj3PJgykGXjvRQ), `code: gcyx`), then place them in the project root directory. 

Please download pre-trained models ([Baidu Disk](https://pan.baidu.com/s/1QbvWxKWccj3PJgykGXjvRQ), `code: gcyx`), and then place the `RealVisXL_V3.0` and `checkpoint` in the project root directory, place the `weights` in `./models`, place the `VQGAN` in `./models/basicsr/experiments_original`.

## Environment
```bash
conda create -n tfg python=3.10  # (Python >= 3.8)
conda activate tfg
pip install -r requirements.txt
```
## Train
```bash
Stage 1: python train.py -opt options/stage1.yml
Stage 2: python train.py -opt options/stage2.yml
```
## Inference
```bash
python test.py
```
