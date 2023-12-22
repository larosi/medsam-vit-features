# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:38:14 2023

@author: Mico
"""


import os
import numpy as np

from skimage.transform import resize
from skimage.color import gray2rgb
import torch
import nibabel as nib
from segment_anything import sam_model_registry
from tqdm import tqdm
import concurrent.futures


def prepare_image(img):
    img = gray2rgb(img)
    img_tensor = resize(img, (1024, 1024))
    img_tensor = img_tensor.transpose((2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = torch.as_tensor(img_tensor, dtype=torch.float32).cuda()
    return img_tensor

def load_model():
    device = torch.cuda.current_device()
    model_path = r'D:\modelzoo\medsam\medsam_vit_b.pth'
    model = sam_model_registry['vit_b'](model_path)
    model = model.to(device)
    model.eval()
    return model

def load_ct(dataset_path, ct_fn):
    ct_path = os.path.join(dataset_path, ct_fn)
    img = nib.load(ct_path)
    data = img.get_fdata()
    spatial_res = img.header.get_zooms()
    return data, spatial_res

def min_max_scale(data):
    return (data - data.min()) / (data.max() - data.min())

def get_dense_descriptor(model, img):
    img_tensor = prepare_image(img)
    features_tensor = model.image_encoder(img_tensor)
    features = features_tensor.cpu().detach().numpy()

    del img_tensor
    del features_tensor
    torch.cuda.empty_cache()

    features = np.squeeze(features)
    return features

def save_features_async(filename, data):
    np.savez_compressed(filename, data)

dataset_path = r'D:\datasets\medseg\medicaldecathlon\Task08_HepaticVessel\Task08_HepaticVessel\imagesTr'

model = load_model()
lower_bound = -160
upper_bound = 240
#https://radiopaedia.org/articles/windowing-ct

for ct_fn in tqdm(os.listdir(dataset_path)):
    features_dir = r'..\data\features\Task08_HepaticVessel'
    features_fn = ct_fn.split('.')[0] + '.npz'
    features_path = os.path.join(features_dir, features_fn)
    if not os.path.exists(features_path):
        ct_data, spatial_res = load_ct(dataset_path, ct_fn)

        ct_data = np.clip(ct_data, lower_bound, upper_bound)
        ct_data = min_max_scale(ct_data)

        all_features = []

        for slice_i in tqdm(range(0, ct_data.shape[2]), desc=ct_fn):
            img = ct_data[:, :, slice_i]
            features = get_dense_descriptor(model, img)
            all_features.append(features)

        features_to_save = np.array(all_features)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(save_features_async, features_path, features_to_save)
