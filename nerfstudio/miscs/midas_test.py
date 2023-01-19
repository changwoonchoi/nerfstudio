# test code for MiDaS depth estimator

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from natsort import natsorted

model_type = "DPT_Large"      # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
scenes = [
    'scene0017_00', 'scene0031_00', 'scene0033_00', 'scene0036_00', 'scene0046_00', 'scene0052_00', 'scene0058_00',
    'scene0059_00', 'scene0068_00', 'scene0072_00'
]
for scene in scenes:
    # root = "/home/ccw/Downloads/scannet_filtered/scene0024_00/color"
    root = os.path.join("/home/ccw/Downloads/scannet_filtered/")
    img_dir = os.path.join(root, scene, "color")
    save_dir = os.path.join(root, scene, "MiDaS_depth")
    os.makedirs(save_dir, exist_ok=True)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device('cuda')
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    for filename in natsorted(glob.glob(os.path.join(img_dir, '*.jpg'))):

        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        output = output / 255.
        output = 1 - output
        plt.imsave(os.path.join(save_dir, filename.split('/')[-1]), output, cmap="viridis")
