#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 06:32:13 2025

@author: smkelley
"""
#%% Load modules

import yaml
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
import preprocess as preproc
import blockaverage as blockavg
import groupaverage as groupavg
import image_recon as img_recon

#%%

#config_path = "/projectnb/nphfnirs/ns/Shannon/Code/cedalion-pipeline/workflow/config/config.yaml" # CHANGE if testing
config_path = "/projectnb/nphfnirs/ns/Shannon/Code/cedalion-pipeline/workflow/scripts/testing/config_test.yaml" # CHANGE if testing

#config_path = "C:\\Users\\shank\\Documents\\GitHub\\cedalion-pipeline\\workflow\\config\\config.yaml"

with open(config_path, 'r') as file:  # open config file
    config = yaml.safe_load(file)
    
cfg_dataset = config['dataset']
cfg_preprocess = config['preprocess']

#%% preproc

subj = "03"  #cfg_dataset['subject'][0]   # sub idx you want to test  # !!! change to loop ?
task = cfg_dataset['task'][0]
run = cfg_dataset['run'][0]

snirf_path = f"{cfg_dataset['root_dir']}/sub-{subj}/nirs/sub-{subj}_task-{task}_run-{run}_nirs.snirf"
events_path =  f"{cfg_dataset['root_dir']}/sub-{subj}/nirs/sub-{subj}_task-{task}_run-{run}_events.tsv"

save_path = f"{cfg_dataset['root_dir']}/derivatives/{cfg_dataset['derivatives_subfolder']}/preprocessed_data/sub-{subj}/"
out_snirf = f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_preprocessed.snirf"
out_json = f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_dataquality.json"

out_files = {
    "out_snirf" : f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_preprocessed.snirf",
    "out_json": f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_dataquality.json",
    "out_dqr": f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_DQR.png",
    "out_gvtd": f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_DQR_gvtd_hist.png",
    "out_slope": f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_slope.png"
    }


der_dir = os.path.join(save_path)
if not os.path.exists(der_dir):
    os.makedirs(der_dir)



preproc.preprocess_func(config, snirf_path, events_path, cfg_dataset, cfg_preprocess, out_files)

