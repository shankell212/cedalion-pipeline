#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 07:39:52 2025

@author: smkelley
"""
import yaml
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
import groupaverage as groupavg

#%%
import importlib
importlib.reload(groupavg)

# config_path = "/projectnb/nphfnirs/ns/Shannon/Code/cedalion-pipeline/workflow/config/config.yaml"
#config_path = "C:\\Users\\shank\\Documents\\GitHub\\cedalion-pipeline\\workflow\\config\\config.yaml"
config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts/testing/config_test_BS.yaml" # CHANGE if testing


with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

cfg_dataset = config['dataset']
cfg_blockaverage = config['blockaverage']
cfg_hrf = config['hrf']
cfg_groupaverage = config['groupaverage']
flag_prune_channels = config['preprocess']['steps']['prune']['enable']
cfg_groupaverage['mse_amp_thresh'] = config['preprocess']['steps']['prune']['amp_thresh']


subjects = cfg_dataset['subject']
task = cfg_dataset['task'][0]
        
blockavg_dir = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "blockaverage")  #, f"sub-{subj}")
blockavg_files = [os.path.join(blockavg_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_blockaverage.pkl") for subj in subjects ]
data_quality_files = [os.path.join(blockavg_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_dataquality.json") for subj in subjects ]
geo_files = [os.path.join(blockavg_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_geo.sidecar") for subj in subjects ]
# blockavg_files_nc = [os.path.join(blockavg_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_blockaverage.nc") for subj in subjects ]
# epoch_files_nc = [os.path.join(blockavg_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_epochs.nc") for subj in subjects ]

# preproc_dir = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "preprocessed_data")  #, f"sub-{subj}")
# data_quality_files = [os.path.join(preproc_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_run-{cfg_dataset['run'][-1]}_nirs_dataquality_geo.sidecar") for subj in subjects ]


save_path = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "groupaverage")
out = os.path.join(save_path, f"task-{task}_nirs_groupaverage.pkl")

der_dir = os.path.join(save_path)
if not os.path.exists(der_dir):
    os.makedirs(der_dir)
        
groupavg.groupaverage_func(cfg_dataset, cfg_blockaverage, cfg_hrf, cfg_groupaverage, flag_prune_channels, blockavg_files, data_quality_files, geo_files, out)


