import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.nn import functional as F
from torchvision import utils, transforms
from pathlib import Path
import PIL
from math import log10, ceil
import numpy as np
import os
import lpips
import pickle
from collections import defaultdict
import sys
sys.path.append('/workspace')
from optim_utils import *
from tqdm import tqdm
from functools import partial
import time
import dnnlib
import contextlib
from facenet_pytorch import MTCNN
from torch.autograd import Variable as V

device = torch.device('cuda:0')

import config_2 as config # update to select different config file

os.makedirs(config.out_folder, exist_ok=True)

ckpt = f'{config.home_dir}/networks/ffhq.pkl'

with open(ckpt, 'rb') as f:
    networks = pickle.Unpickler(f).load()

G_inset = networks['G_ema'].to(device)
print("G_inset:")
print(f"  Type: {type(G_inset)}")

# define average face
# w_samples = G_inset.mapping(torch.from_numpy(np.random.RandomState(123).randn(10000, G_inset.z_dim)).to(device), None)
# w_samples = w_samples[:, :1, :]
# latent_avg_inset = torch.mean(w_samples, axis=0).squeeze()
# latent_avg_inset = latent_avg_inset.unsqueeze(0).repeat(G_inset.num_ws, 1).unsqueeze(0)

print('... loaded inset generator.')


def print_shapes_and_types(G_canvas):
    # Step 1: Generate random latent vectors (Z-space)
    z_samples = torch.from_numpy(np.random.RandomState(123).randn(10000, G_canvas.z_dim)).to(device)
    print("z_samples:")
    print(f"  Type: {type(z_samples)}")
    print(f"  Shape: {z_samples.shape}\n")

    # Step 2: Map Z-space to W-space
    w_samples = G_canvas.mapping(z_samples, None)
    print("w_samples:")
    print(f"  Type: {type(w_samples)}")
    print(f"  Shape: {w_samples.shape}\n")

    # Step 3: Select the first W vector and calculate the average latent vector
    w_samples_selected = w_samples[:, :1, :]
    latent_avg_inset = torch.mean(w_samples_selected, axis=0).squeeze()
    print("w_samples_selected:")
    print(f"  Type: {type(w_samples_selected)}")
    print(f"  Shape: {w_samples_selected.shape}\n")

    print("latent_avg_canvas:")
    print(f"  Type: {type(latent_avg_inset)}")
    print(f"  Shape: {latent_avg_inset.shape}\n")

    # Step 4: Repeat latent_avg_canvas and reshape
    latent_avg_inset_expanded = latent_avg_inset.unsqueeze(0).repeat(G_canvas.num_ws, 1).unsqueeze(0)
    print("latent_avg_canvas_expanded:")
    print(f"  Type: {type(latent_avg_inset_expanded)}")
    print(f"  Shape: {latent_avg_inset_expanded.shape}\n")

    return latent_avg_inset_expanded


# Assuming G_canvas is loaded and initialized
latent_avg_inset = print_shapes_and_types(G_inset)

z = torch.from_numpy(np.random.RandomState(config.seed_inset).randn(32, G_inset.z_dim)).to(device)
with torch.no_grad():
    random_face_w = G_inset.mapping(z, None)
    print("random_face_w:")
    print(f"  Type: {type(random_face_w)}")
    print(f"  Shape: {random_face_w.shape}\n")
    if hasattr(config, 'trunc_insets'):
        for i in range(18):
            random_face_w[:, i, :] = random_face_w[:, i, :] * (1 - config.trunc_insets[i]) + latent_avg_inset[:, i, :] * config.trunc_insets[i]
    elif hasattr(config, 'trunc_inset'):
        random_face_w = random_face_w * (1 - config.trunc_inset) + latent_avg_inset * config.trunc_inset
    random_outputs = G_inset.synthesis(random_face_w.to(device), noise_mode='const')
save_tensor(random_outputs, 'face', out_folder=config.out_folder)

faces  = config.selected_faces if hasattr(config, 'selected_faces') else range(len(random_face_w))
print("faces:")
print(f"  content: {faces}\n")
print(f"  Type: {type(faces)}")
print(f"  Len: {len(faces)}\n")

for idx in range(len(faces)):
    face = faces[idx]

    latent_w_inset = random_face_w[face].unsqueeze(0).detach().clone()
    print("latent_w_inset:")
    print(f"  content: {latent_w_inset}")
    print(f"  Type: {type(latent_w_inset)}")
    print(f"  Shape: {len(latent_w_inset.shape)}\n")


