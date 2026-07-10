# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:43:02 2026

@author: codett
"""

import numpy as np

def normalize_data(x):
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    return x_min, x_max, (x - x_min) / (x_max - x_min + 1e-8)

def load_ampds_data(ampds_filepath, T_limit):
    
    data = np.load(ampds_filepath)
    x, y = data['X'], data['Y']
    appliance_names = data['out_labels']
    T = x.shape[0]
    T_limit = min(T, T_limit) if T_limit is not None else T
    x, y = x[:T_limit], y[:T_limit]
    x = x[:, [0,2]] # For X data, keep only columns corresponding to P and Q
    y = y[:,:,0] # For Y data, keep only column corresponding to P
    
    # Normalize
    x_min, x_max, x_norm = normalize_data(x)
    y_min, y_max, y_norm = normalize_data(y)
    scaling_factors = {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max}
    
    return {
        'x': x,
        'x_norm': x_norm,
        'y': y,
        'y_norm': y_norm,
        'scaling_factors': scaling_factors,
        'appliance_names': appliance_names,
        'num_timesteps': T_limit}