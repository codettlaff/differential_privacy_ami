# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 12:27:39 2026

@author: codett
"""
import os
import tqdm
import numpy as np
import pickle
import sys

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
AMPDS_FILEPATH = os.path.join(DATA_DIR, 'ampds2.npz')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

SRC_DIR = os.path.join(BASE_DIR, 'src')
sys.path.append(SRC_DIR)
import differential_privacy as dp
import load_data
import nilm

MODEL_NAME = 'nilm_cnn_model.keras'
MODEL_FILEPATH = os.path.join(RESULTS_DIR, MODEL_NAME)

EXPERIMENT_NAME = 'privacy_gaurantee_experiment'
EXPERIMENT_RESULT_FILEPATH = os.path.join(RESULTS_DIR, EXPERIMENT_NAME + '.pkl')

# NILM Training Settings
T_LIMIT = 86400 # Two Months for AMPDS Data
TRAIN_SPLIT = [0.7, 0.15, 0.15] # Train Test Val Split
WINDOW_LENGTH, STRIDE = 30, 1
EPOCHS = 20
BATCH_SIZE = 32

# Differential Privacy Settings
B = 5e3 # Appliance Sensitivity
EPSILON_VALUES = np.linspace(0.1, 1e3, 10) # Epsilon Range

# Other Experimental Settings
M = 20 # Monte Carlo Number of Trials

def experiment(do_train_model=False):
    
    # Load Data
    data = load_data.load_ampds_data(AMPDS_FILEPATH, T_LIMIT)
    x = data['x']
    scaling_factors = data['scaling_factors']
    num_timesteps = data['num_timesteps']
    
    # Precompute Indices for Windowing
    num_windows, indices = nilm.precompute_indices(num_timesteps, WINDOW_LENGTH, STRIDE, TRAIN_SPLIT)
    
    # Train Model
    if do_train_model: nilm.train_model(data, indices, WINDOW_LENGTH, EPOCHS, BATCH_SIZE, MODEL_FILEPATH)
    
    # Get Baseline Results
    baseline_results = nilm.test_model(MODEL_FILEPATH, data, indices['test'], WINDOW_LENGTH, BATCH_SIZE, scaling_factors, show=False)

    # Epsilon Trials
    eacc = {}
    eacc['baseline'] = baseline_results['eacc']
    for epsilon in tqdm(EPSILON_VALUES, desc="Epsilon Loop"):
        eacc_trials = []
        for m in tqdm(range(M), desc=f"M loop (eps={epsilon})", leave=False):
            x_tilde = x
            x_tilde[:,0] = dp.make_private_load_profile(B, epsilon, x[:,0]) # Add noise to real power only.
            x_tilde_min, x_tilde_max, x_tilde_norm = load_data.normalize_data(x_tilde)
            data['x'], data['x_norm'] = x_tilde, x_tilde_norm
            scaling_factors['x_min', 'x_max'] = x_tilde_min, x_tilde_max
            results = nilm.test_model(MODEL_FILEPATH, data, indices['test'], WINDOW_LENGTH, BATCH_SIZE, scaling_factors, show=False)
            eacc_trials.append(results['eacc'])
        eacc_trials = np.array(eacc_trials)
        eacc[epsilon] = np.mean(eacc_trials, axis=0)
            
    
    with open(EXPERIMENT_RESULT_FILEPATH, 'wb') as f: pickle.dump(eacc, f)
    
def display_results():
    
    with open(EXPERIMENT_RESULT_FILEPATH, 'wb') as f: eacc = pickle.load(f)
    
    print(f"{'Appliance':<12} {'Baseline':>10}", end="")
    for epsilon in EPSILON_VALUES: print(f" {epsilon:>10}", end="")
    print()
    
    keys = list(eacc.keys())
    epsilon_values = keys
    epsilon_values.remove('appliance_names')
    epsilon_values.remove('baseline')
    appliance_names = keys['appliance_names']
    
    for i, appliance in enumerate(appliance_names):
        print(f"{appliance:<12} {eacc['baseline'][i]:10.4f}", end="")
        for epsilon in epsilon_values: print(f" {eacc[epsilon][i]:10.4f}", end="")
        print()
        
if __name__ == '__main__':
    
    experiment(do_train_model=True)
    display_results()