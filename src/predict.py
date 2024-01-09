# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 11:20:04 2023

@author: Mico
"""

import os
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objs as go
from joblib import load

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from prepare_labels import load_ct, get_labelmap, imshow_labels
from train_fine_grain import SimpleTransformer, CT3DDataset, df_to_lowres
from train_fine_grain import load_features, create_dataframe, min_max_scale, normalize_coordinates
from joblib import load

features_dir = r'..\data\features\Task08_HepaticVessel'
labels_dir = r'..\data\labels\Task08_HepaticVessel'
model_dir = os.path.join('..', 'models', 'medsam_fine_grain_3d')
#model_path = os.path.join(model_dir, 'dice_loss', 'model_best.pth')
model_path = os.path.join(model_dir, 'model_epoch_0009.pth')

df_loss = pd.read_csv(os.path.join(model_dir, 'dice_loss', 'losses.csv'))
train_files = list(df_loss['ct'].unique())

features_dir = r'..\data\features\Task08_HepaticVessel'
labels_dir = r'..\data\labels\Task08_HepaticVessel'
dataset_dir = r'D:\datasets\medseg\medicaldecathlon\Task08_HepaticVessel\Task08_HepaticVessel\imagesTr'

ct_filenames = os.listdir(labels_dir)

# validation dataset unused in training
test_files = list(set(ct_filenames) - set(train_files))


labelmap, labelmap_inv = get_labelmap()
label_encoder = OneHotEncoder(handle_unknown='ignore')
label_encoder.fit(np.array(list(labelmap.keys())).reshape(-1, 1))

pca = load(r'..\models\pca\pca_63.joblib')

device = torch.cuda.current_device()
model = SimpleTransformer(input_dim=64, num_heads=8, num_classes=len(labelmap))
model.load(model_path)
model.to(device)
num_points = 64*64*8
lower_bound = -160
upper_bound = 240

lowres_prediction = True
for ct_fn in tqdm(test_files[1:2]):
    features_fn = ct_fn.split('.')[0] + '.npz'
    features_path = os.path.join(features_dir, features_fn)
    labels_path = os.path.join(labels_dir, ct_fn)

    all_features = load_features(features_path, pca)
    ct_data, spatial_res = load_ct(dataset_dir, ct_fn)
    invalid_mask = np.logical_and(ct_data < lower_bound, ct_data > upper_bound)
    ct_data = np.clip(ct_data, lower_bound, upper_bound)
    ct_data = min_max_scale(ct_data)
                
    ct_labels, spatial_res = load_ct(labels_dir, ct_fn)
    df = create_dataframe(ct_labels, ct_data, spatial_res, all_features, division_factor=8)
    df = normalize_coordinates(df)
    if lowres_prediction:
        df = df_to_lowres(df)
    all_features = all_features.reshape(-1, all_features.shape[-1]).astype(np.float32)

    batch_size = int(np.ceil(len(df) / (len(df) // num_points)))

    test_dataset = CT3DDataset(df, all_features, label_encoder, batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    predictions = []
    for features_batch, labels_batch in tqdm(test_loader, leave=False):
        features_batch = features_batch.to(device)
        labels_batch = torch.squeeze(labels_batch).to(device)
        with torch.no_grad():
            logits = model(features_batch)
            predictions.append(F.softmax(logits, dim=1).cpu().detach().numpy())
    
    
    h, w, d = ct_data.shape
    if lowres_prediction:
        h, w = 64, 64
    prediction = np.concatenate(predictions, axis=0)
    
    y_prob = prediction.reshape((h, w, d, len(labelmap)))
    y_pred = np.argmax(y_prob, axis=-1)
    
    imshow_labels(y_pred)
    
    from skimage import io
    for slice_i in tqdm(range(0, y_prob.shape[2])):
        img = y_prob[:,:,slice_i, 1:4]
        io.imshow(img)
        io.show()

    imshow_labels(ct_labels.astype(int))
    
