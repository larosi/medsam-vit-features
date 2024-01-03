# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 12:19:34 2024

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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from prepare_labels import load_ct, get_labelmap

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_classes):
        super(SimpleTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.view(-1, x.size(-1))
        return self.classifier(x)
    
    def save_checkpoint(self, save_dir, epoch):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        epoch_str = str(epoch).zfill(4)
        model_path = os.path.join(save_dir, f'model_epoch_{epoch_str}.pth')
        self.save(model_path)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(model_path, map_location=device))


class CT3DDataset(Dataset):
    def __init__(self, dataframe, all_features, label_encoder, foreground_mean, foreground_std, batch_size):
        self.dataframe = dataframe
        self.all_features = all_features
        self.label_encoder = label_encoder
        self.foreground_mean = foreground_mean
        self.foreground_std = foreground_std
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataframe) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        df_batch = self.dataframe.iloc[start:end]
        patch_id = df_batch['patch_id'].values
        xyz = df_batch[['x', 'y', 'z']].values
        xyz = (xyz - self.foreground_mean) / (2.5 * self.foreground_std[1])
        pe = positional_encoding_3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], D=256, scale=10)
        positional_norm = np.linalg.norm(pe) / 2
        features = self.all_features[patch_id] + pe / positional_norm
        
        label = df_batch['label'].values
        label = np.expand_dims(label, axis=-1)
        label = self.label_encoder.transform(label).toarray()

        return torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(label, dtype=torch.float32)

def positional_encoding_3d(x, y, z, D, scale=10):
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    n_points = x.shape[0]
    encoding = np.zeros((n_points, D))

    for i in range(D // 6):
        encoding[:, 2*i] = np.sin(x / (scale ** (6 * i / D)))
        encoding[:, 2*i + 1] = np.cos(x / (scale ** (6 * i / D)))
        encoding[:, 2*i + D // 3] = np.sin(y / (scale ** (6 * i / D)))
        encoding[:, 2*i + 1 + D // 3] = np.cos(y / (scale ** (6 * i / D)))
        encoding[:, 2*i + 2 * D // 3] = np.sin(z / (scale ** (6 * i / D)))
        encoding[:, 2*i + 1 + 2 * D // 3] = np.cos(z / (scale ** (6 * i / D)))

    return encoding


def load_features(features_path):
    with np.load(features_path) as data:
        all_features = np.load(features_path)['arr_0']
    all_features = np.transpose(all_features, (2, 3, 0, 1))
    return all_features

def create_dataframe(ct_labels, spatial_res, all_features, division_factor=8):
    x, y, z = np.meshgrid(np.arange(0, ct_labels.shape[0]),
                          np.arange(0, ct_labels.shape[1]),
                          np.arange(0, ct_labels.shape[2]))

    # super voxel id
    block_size = np.ceil(np.array(ct_labels.shape) / division_factor).astype(int)
    block_ids = (x // block_size[0]) + (y // block_size[1]) * division_factor + (z // block_size[2]) * division_factor**2

    # features patch id
    all_features_ids = np.arange(np.prod(all_features.shape[:-1]))
    all_features_ids = np.expand_dims(all_features_ids, axis=-1)
    all_features_ids = all_features_ids.reshape(all_features.shape[:-1])
    all_features_ids = zoom(all_features_ids, (8, 8, 1), order=0)

    # save metrics coordinates, ids and labels into a dataframe
    df = pd.DataFrame()
    df['x'] = x.flatten() * spatial_res[0]
    df['y'] = y.flatten() * spatial_res[1]
    df['z'] = z.flatten() * spatial_res[2]
    df['label'] = ct_labels.flatten().astype(int)
    df['patch_id'] = all_features_ids.flatten().astype(int)
    df['block_id'] = block_ids.flatten().astype(int)
    return df


def plot_loss_metrics(df_loss):
    avg_loss_per_epoch = df_loss.groupby('epoch').agg({'train_loss': ['mean', 'std'],
                                                       'test_loss': ['mean', 'std']}).reset_index()
    avg_loss_per_epoch.columns = ['epoch', 'train_loss_mean', 'train_loss_std', 'test_loss_mean', 'test_loss_std']

    fig = go.Figure()

    colors = {'train': 'blue', 'test': 'red'}
    colors_rgba = {'train': 'rgba(68, 68, 128, 0.2)',
                   'test': 'rgba(128, 68, 68, 0.2)'}

    for split in ['train', 'test']:
        fig.add_trace(go.Scatter(
            name=f'{split}',
            x=avg_loss_per_epoch['epoch'],
            y=avg_loss_per_epoch[f'{split}_loss_mean'],
            mode='markers+lines',
            line=dict(color=colors[split]),
        ))
        for sign in [-1, 1]:
            if sign == -1:
                name = f'{split} - std'
            else:
                name = f'{split} + std'
            fig.add_trace(go.Scatter(
                name=name,
                x=avg_loss_per_epoch['epoch'],
                y=avg_loss_per_epoch[f'{split}_loss_mean'] + sign * avg_loss_per_epoch[f'{split}_loss_std'],
                mode='lines',
                fillcolor=colors_rgba[split],
                fill='tonexty',
                marker=dict(color=colors_rgba[split]),
                line=dict(width=0),
                showlegend=False
            ))

    fig.update_layout(
        yaxis_title='Loss',
        title='Train Loss per Epoch with Batch Standard Desviation',
        hovermode="x"
    )
    return fig


if __name__ == '__main__':
    features_dir = r'..\data\features\Task08_HepaticVessel'
    labels_dir = r'..\data\labels\Task08_HepaticVessel'
    save_dir = os.path.join('..', 'models', 'medsam_fine_grain_3d')

    labelmap = get_labelmap()
    label_encoder = OneHotEncoder(handle_unknown='ignore')
    label_encoder.fit(np.array(list(labelmap.keys())).reshape(-1, 1))

    device = torch.cuda.current_device()
    model = SimpleTransformer(input_dim=256, num_heads=4, num_classes=len(labelmap))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_points = 64*64
    num_epochs = 10

    train_metrics = {'epoch': [],
                     'train_loss': [],
                     'test_loss': [],
                     'ct': []}

    ct_filenames = os.listdir(labels_dir)
    for epoch in tqdm(range(num_epochs), position=0, desc='epoch'):
        with tqdm(total=len(ct_filenames), desc='batch', position=1) as batch_pbar:
            for ct_fn in ct_filenames:
                features_fn = ct_fn.split('.')[0] + '.npz'
                features_path = os.path.join(features_dir, features_fn)
                labels_path = os.path.join(labels_dir, ct_fn)

                all_features = load_features(features_path)

                ct_labels, spatial_res = load_ct(labels_dir, ct_fn)
                df = create_dataframe(ct_labels, spatial_res, all_features, division_factor=8)
                all_features = all_features.reshape(-1, all_features.shape[-1]).astype(np.float32)

                foreground_stats = df[df['label'] > 0][['x', 'y', 'z']].agg(['mean', 'std'])
                foreground_mean = foreground_stats.loc['mean'].values
                foreground_std = foreground_stats.loc['std'].values

                df_sample = df.groupby(['label'], group_keys=False).apply(lambda x: x.sample(min(len(x), 15000)))
                # df_sample.to_csv(r'..\data\ct_sample.txt', sep=' ', index=False)
                df_train, df_test, _, _ = train_test_split(df_sample, df_sample['label'],
                                                           stratify=df_sample['label'],
                                                           random_state=42, train_size=0.7)
                df_train.sort_values(by='block_id', inplace=True)
                df_test.sort_values(by='block_id', inplace=True)

                train_batch_size = int(np.ceil(len(df_train) / (len(df_train) // num_points)))
                test_batch_size = int(np.ceil(len(df_test) / (len(df_test) // num_points)))

                train_dataset = CT3DDataset(df_train, all_features, label_encoder,
                                            foreground_mean, foreground_std, train_batch_size)
                test_dataset = CT3DDataset(df_test, all_features, label_encoder,
                                           foreground_mean, foreground_std, test_batch_size)

                train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                total_train_loss = 0
                total_test_loss = 0

                # train loop
                model.train()
                for features_batch, labels_batch in train_loader:
                    features_batch = features_batch.to(device)
                    labels_batch = torch.squeeze(labels_batch).to(device)
                    optimizer.zero_grad()
                    outputs = model(features_batch)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()

                # test loop
                for features_batch, labels_batch in test_loader:
                    features_batch = features_batch.to(device)
                    labels_batch = torch.squeeze(labels_batch).to(device)
                    outputs = model(features_batch)
                    loss = criterion(outputs, labels_batch)
                    total_test_loss += loss.item()

                avg_train_loss = total_train_loss / len(train_loader)
                avg_test_loss = total_test_loss / len(test_loader)

                batch_pbar.set_postfix({'Train Loss': avg_train_loss, 'Test Loss': avg_test_loss})
                batch_pbar.update()

                train_metrics['epoch'].append(epoch)
                train_metrics['train_loss'].append(avg_train_loss)
                train_metrics['test_loss'].append(avg_test_loss)
                train_metrics['ct'].append(ct_fn)

            # Save train and test loss metrics
            model.save_checkpoint(save_dir, epoch)
            df_loss = pd.DataFrame(train_metrics)
            fig = plot_loss_metrics(df_loss)
            fig.write_html(os.path.join(save_dir, 'losses.html'))
            df_loss.to_csv(os.path.join(save_dir, 'losses.csv'), index=False)
