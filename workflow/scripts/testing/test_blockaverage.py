#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 07:37:37 2025

@author: smkelley
"""


import yaml
import os
import sys
sys.path.append('/projectnb/nphfnirs/ns/Shannon/Code/cedalion-pipeline/workflow/scripts/')
import blockaverage as blockavg


#%%

config_path = "/projectnb/nphfnirs/ns/Shannon/Code/cedalion-pipeline/workflow/config/config.yaml"
#config_path = "C:\\Users\\shank\\Documents\\GitHub\\cedalion-pipeline\\workflow\\config\\config.yaml"  # change if debugging

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

cfg_dataset = config['dataset']
cfg_blockaverage = config['blockaverage']
cfg_hrf = config['hrf']

subj = cfg_dataset['subject'][0]   # sub idx you want to test
task = cfg_dataset['task'][0]
run = cfg_dataset['run']

preproc_dir = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "preprocessed_data")

run_files = [os.path.join(preproc_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_run-{r}_nirs_preprocessed.snirf") for r in run]

data_quality_files = [os.path.join(preproc_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_run-{r}_nirs_dataquality.json") for r in run]

# run_files = ["/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/derivatives/cedalion/preprocessed_data/sub-01/sub-01_task-STS_run-01_nirs_preprocessed.snirf"]

save_path = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "blockaverage", f"sub-{subj}")
out_pkl = os.path.join(save_path,f"sub-{subj}_task-{task}_nirs_blockaverage.pkl")
out_json = os.path.join(save_path,f"sub-{subj}_task-{task}_nirs_dataquality.json")
out_blkavg_nc = os.path.join(save_path, f"sub-{subj}_task-{task}_nirs_blockaverage.nc")
out_epoch_nc = os.path.join(save_path, f"sub-{subj}_task-{task}_nirs_epochs.nc")

der_dir = os.path.join(save_path)
if not os.path.exists(der_dir):
    os.makedirs(der_dir)
        
blockavg.blockaverage_func(cfg_dataset, cfg_blockaverage, cfg_hrf, run_files, data_quality_files, out_pkl, out_json, out_blkavg_nc, out_epoch_nc)

