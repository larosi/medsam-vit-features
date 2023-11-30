# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:38:14 2023

@author: Mico
"""


import os
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.color import gray2rgb
import torch
import nibabel as nib
from segment_anything import sam_model_registry
from tqdm import tqdm


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


dataset_path = r'D:\datasets\medseg\medicaldecathlon\Task08_HepaticVessel\Task08_HepaticVessel\imagesTr'
dataset_label_path = r'D:\datasets\medseg\medicaldecathlon\Task08_HepaticVessel\Task08_HepaticVessel\labelsTr'

features_dir = r'..\data\features'
ct_fn = 'hepaticvessel_001.nii.gz'

ct_data, spatial_res = load_ct(dataset_path, ct_fn)
ct_labels, spatial_res = load_ct(dataset_label_path, ct_fn)

target_tissues = ct_data[ct_labels > 0]

target_min = target_tissues.min()
target_max = target_tissues.max()
target_std = target_tissues.std()
target_mean = target_tissues.mean()

print('Vessels HU ranges')
print(f'mean {target_mean}')
print(f'std {target_std}')
print(f'min-max {target_min} {target_max}')

lower_bound = -100
upper_bound = 400
ct_data = np.clip(ct_data, lower_bound, upper_bound)
#https://radiopaedia.org/articles/windowing-ct
ct_data = min_max_scale(ct_data)

"""
for slice_i in tqdm(range(0, ct_data.shape[2])):
    im_labels = ct_data[:, :, slice_i]
    
    io.imshow(im_labels)
    io.show()
"""
model = load_model()
all_features = []

for slice_i in tqdm(range(0, ct_data.shape[2])):
    img = ct_data[:, :, slice_i]
    features = get_dense_descriptor(model, img)
    all_features.append(features)

all_features = np.array(all_features)

features_fn = ct_fn.split('.')[0] + '.npy'
features_path = os.path.join(features_dir, features_fn)
np.save(features_path, all_features)
