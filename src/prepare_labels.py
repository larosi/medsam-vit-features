# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:54:04 2024

@author: Mico
"""

import os
import numpy as np
from skimage import io
import nibabel as nib
from tqdm import tqdm

def save_ct(dataset_path, ct_fn, data, spatial_res):
    ct_path = os.path.join(dataset_path, ct_fn)
    affine = np.diag(spatial_res + (1,))
    img = nib.Nifti1Image(data, affine)
    img.to_filename(ct_path)

def load_ct(dataset_path, ct_fn):
    ct_path = os.path.join(dataset_path, ct_fn)
    img = nib.load(ct_path)
    data = img.get_fdata()
    spatial_res = img.header.get_zooms()
    return data, spatial_res

def load_abdomen_atlas_labels(ct_fn):
    ds_path = r'D:\datasets\AbdomenAtlas\AbdomenAtlas\10_Decathlon_'
    ds_path = ds_path + ct_fn.split('.')[0]
    ds_path = os.path.join(ds_path, 'segmentations')

    segmentation_files = [fn for fn in os.listdir(ds_path) if '.nii' in fn]
    segmentation_files = [fn for fn in segmentation_files if '._' not in fn]
    label_dict = {}
    for fn in segmentation_files:
        label_name = fn.split('.')[0]
        ct_labels, spatial_res = load_ct(ds_path, fn)
        label_dict[label_name] = ct_labels > 0
    return label_dict

def imshow_labels(ct_multilabel):
    colors = [[0, 0, 0],
              [31, 119, 180],
              [255, 127, 14],
              [44, 160, 44],
              [214, 39, 40],
              [148, 103, 189],
              [140, 86, 75],
              [227, 119, 194],
              [127, 127, 127],
              [188, 189, 34],
              [23, 190, 207],
              [255, 255, 255]]
    colors = np.array(colors, dtype=np.uint8)
    for slice_i in range(0, ct_multilabel.shape[2]):
        img = ct_multilabel[:, :, slice_i]
        img_rgb = colors[img]
        io.imshow(img_rgb)
        io.show()

dataset_path = r'D:\datasets\medseg\medicaldecathlon\Task08_HepaticVessel\Task08_HepaticVessel\imagesTr'
dataset_label_path = r'D:\datasets\medseg\medicaldecathlon\Task08_HepaticVessel\Task08_HepaticVessel\labelsTr'
label_output_dir = os.path.join('..', 'data', 'labels', 'Task08_HepaticVessel')

def get_labelmap():
    label_names = ['background',
                   'liver',
                   'liver_vessel',
                   'liver_tumor',
                   'aorta',
                   'gall_bladder',
                   'kidney_left',
                   'kidney_right',
                   'pancreas',
                   'postcava',
                   'spleen',
                   'stomach']

    labelmap = dict(zip(np.arange(0, len(label_names)), label_names))
    return labelmap


if __name__ == '__main__':
    ct_filenames = os.listdir(dataset_path)
    labelmap = get_labelmap()
    for ct_fn in tqdm(ct_filenames):
        #ct_data, spatial_res = load_ct(dataset_path, ct_fn)
        ct_labels_fine_grain, spatial_res = load_ct(dataset_label_path, ct_fn)
        ct_labels_fine_grain = ct_labels_fine_grain.astype(int)  
        ct_labels_atlas = load_abdomen_atlas_labels(ct_fn)

        ct_multilabel = np.zeros(ct_labels_atlas['liver'].shape, dtype=int)

        # label atlas
        for label_id, label_name in labelmap.items():
            if label_name in ct_labels_atlas.keys():
                mask = ct_labels_atlas[label_name]
                ct_multilabel[mask] = label_id

        # fine_grain labels
        for fine_id in [1, 2]:
            mask = ct_labels_fine_grain == fine_id
            ct_multilabel[mask] = fine_id + 1

        save_ct(label_output_dir, ct_fn, ct_multilabel, spatial_res)
        #imshow_labels(ct_multilabel)
