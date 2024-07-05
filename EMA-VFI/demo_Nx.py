import cv2
import math
import sys
import torch
import os
import numpy as np
import argparse
from imageio import mimsave

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours_t', type=str)
parser.add_argument('--n', default=16, type=int)
args = parser.parse_args()
assert args.model in ['ours_t', 'ours_small_t'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small_t':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
model = Model(-1)
model.load_model()
model.eval()
model.device()


print(f'=========================Start Generating=========================')
image_dir = '/content/drive/MyDrive/gaussian_blurring/3/blurred_frames_kernel8'

# 이미지 로드 및 준비
image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
num_images = len(image_files)
images = []

for i in range(num_images - 1): # 20, 50
    I0 = cv2.imread(image_files[i])
    I2 = cv2.imread(image_files[i + 1])

    images.append(I0[:, :, ::-1])

    I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

    padder = InputPadder(I0_.shape, divisor=32)
    I0_, I2_ = padder.pad(I0_, I2_)

    preds = model.multi_inference(I0_, I2_, TTA=TTA, time_list=[(i+1)*(1./args.n) for i in range(args.n - 1)], fast_TTA=TTA)

    for pred in preds:
        images.append((padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1])

# 마지막 이미지 추가
images.append(cv2.imread(image_files[-1])[:, :, ::-1]) # image_files[50] 

# MP4로 저장
# video_path = f'{image_dir}.mp4'
video_path = '/content/drive/MyDrive/0704/3_cut_2x/3_output_8.mp4'
height, width, layers = images[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
video = cv2.VideoWriter(video_path, fourcc, 60, (width, height))

for image in images:
    video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

video.release()
for i, image in enumerate(images):
    print(f"Shape of image {i}: {image.shape}")
print(f'=========================Done=========================')
