#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 10:14:25 2025

@author: smkelley
"""

#%% Imports
import cedalion
from cedalion.physunits import units

import os
import yaml
import pickle 
import gzip 
import json

import numpy as np
import xarray as xr

#%% Define RMSE func
def compute_rmse(da_gt: xr.DataArray, da_test: xr.DataArray) -> xr.DataArray:
    """Compute RMSE between two xarray DataArrays with dims (chromo, channel, time)."""
    squared_error = (da_gt - da_test) ** 2
    if 'time' in squared_error.dims:
        mse = squared_error.mean(dim='time')
    else:
        mse = squared_error.mean(dim='reltime')
    rmse = np.sqrt(mse)
    return rmse  # dims: (chromo, channel)


#%% Load in data
# grab 1 subject file
subj_idx = 0 # CHANGE
task_idx = 0
run_idx = 0


#root_dir = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts/testing/regression_testing"
root_dir = "/projectnb/nphfnirs/s/users/shannon/Data/reg_test_data"

# load in config
root_dir_config = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts/testing/regression_testing"
config_path = os.path.join(root_dir_config, 'config_BS_reg_test.yml')


with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

cfg_dataset = config['dataset']
cfg_blockavg = config['blockaverage']
subject = cfg_dataset['subject']
task = cfg_dataset['task']
run = cfg_dataset['run']

#-----
# Load in ground_truth data
#-----
gr_truth_results_dir = os.path.join(root_dir, 'ground_truth')

preproc_dir = os.path.join(gr_truth_results_dir, 'preprocessed_data')
preproc_files = [[[os.path.join(f'sub-{s}',f'sub-{s}_task-{t}_run-{r}_nirs_preprocessed.pkl') for r in run] for t in task] for s in subject]
dataqual_files = [[[os.path.join(f'sub-{s}',f'sub-{s}_task-{t}_run-{r}_nirs_dataquality_geo.sidecar') for r in run] for t in task] for s in subject]


preproc_singlefile = preproc_files[subj_idx][task_idx][run_idx]
dataqual_singlefile = dataqual_files[subj_idx][task_idx][run_idx]  

with gzip.open(os.path.join(preproc_dir, preproc_singlefile), 'rb') as f:  # open pkl
    record = pickle.load(f)
    rec_gt = record[0]

with gzip.open(os.path.join(preproc_dir, dataqual_singlefile), 'rb') as f:  # open dataqual sidecar
    data_qual_gt = pickle.load(f)


# load in group average data
groupavg_dir = os.path.join(gr_truth_results_dir, 'groupaverage')
groupavg_files = f'task-{task[0]}_nirs_groupaverage_{cfg_blockavg["rec_str"]}.pkl'

# Load in data
groupaverage_path = os.path.join(groupavg_dir, groupavg_files)
if os.path.exists(groupaverage_path):
    with open(groupaverage_path, 'rb') as f:
        groupavg_results_gt = pickle.load(f)
  
    blockaverage_weighted_gt = groupavg_results_gt['group_blockaverage_weighted']  #groupavg_results['group_blockaverage_weighted']
    # blockaverage = groupavg_results['group_blockaverage']
    # blockaverage_stderr = groupavg_results['total_stderr_blockaverage']
    # blockaverage_subj = groupavg_results['blockaverage_subj']
    # blockaverage_mse_subj = groupavg_results['blockaverage_mse_subj']
    print(f" {groupaverage_path} loaded successfully!")

else:
    print(f"Error: File '{groupaverage_path}' not found!")


     
# -----
# Load in test data 
# -----
test_results_dir = os.path.join(root_dir, 'test_data', 'derivatives', cfg_dataset['derivatives_subfolder'])

preproc_dir = os.path.join(test_results_dir, 'preprocessed_data')
preproc_files = [[[os.path.join(f'sub-{s}',f'sub-{s}_task-{t}_run-{r}_nirs_preprocessed.pkl') for r in run] for t in task] for s in subject]
dataqual_files = [[[os.path.join(f'sub-{s}',f'sub-{s}_task-{t}_run-{r}_nirs_dataquality_geo.sidecar') for r in run] for t in task] for s in subject]

preproc_singlefile = preproc_files[subj_idx][task_idx][run_idx]
dataqual_singlefile = dataqual_files[subj_idx][task_idx][run_idx]  

with gzip.open(os.path.join(preproc_dir, preproc_singlefile), 'rb') as f:  # open pkl
    record = pickle.load(f)
    rec_test = record[0]
    
with gzip.open(os.path.join(preproc_dir, dataqual_singlefile), 'rb') as f:  # open dataqual sidecar
    data_qual_test = pickle.load(f)

# load in group average data
groupavg_dir = os.path.join(test_results_dir, 'groupaverage')
groupavg_files = f'task-{task[0]}_nirs_groupaverage_{cfg_blockavg["rec_str"]}.pkl'


# Load in data
groupaverage_path = os.path.join(groupavg_dir, groupavg_files)
if os.path.exists(groupaverage_path):
    with open(groupaverage_path, 'rb') as f:
        groupavg_results_test = pickle.load(f)
  
    blockaverage_weighted_test = groupavg_results_test['group_blockaverage_weighted']  #groupavg_results['group_blockaverage_weighted']
    # blockaverage = groupavg_results['group_blockaverage']
    # blockaverage_stderr = groupavg_results['total_stderr_blockaverage']
    # blockaverage_subj = groupavg_results['blockaverage_subj']
    # blockaverage_mse_subj = groupavg_results['blockaverage_mse_subj']
    # print(f" {groupaverage_path} loaded successfully!")

else:
    print(f"Error: File '{groupaverage_path}' not found!")


#%% Perform regression test using RMSE

# timeseries
print('regression test on conc timeseries: ')
conc_gt = rec_gt['conc']
conc_test = rec_test['conc']

rmse = compute_rmse(conc_gt, conc_test) # Compute RMSE
mean_rmse_per_chromo = rmse.mean(dim='channel')

threshold = 0.01*units.micromolar  # micromolar
failing_channels = (rmse > threshold).sum(dim='channel')

print(f"Channels failing threshold per chromo:\n{failing_channels}")
assert (rmse < threshold).all(), "Regression test failed: RMSE too high."

# grp avg
print('\nregression test on weighted group average results: ')

rmse_grp = compute_rmse(blockaverage_weighted_gt, blockaverage_weighted_test) # Compute RMSE
mean_rmse_per_chromo_grp = rmse.mean(dim='channel')

if cfg_blockavg["rec_str"] == 'conc':
    threshold = 0.01*units.micromolar  
else:
    threshold = 0.01
failing_channels_grp = (rmse_grp > threshold).sum(dim='channel')

print(f"Channels failing threshold per chromo:\n{failing_channels}")
assert (rmse_grp < threshold).all(), "Regression test failed: RMSE too high."


#%%


