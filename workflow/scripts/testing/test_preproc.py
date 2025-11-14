#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 07:23:40 2025

@author: smkelley
"""

#%% Imports
import yaml
import os
import copy
import sys
# script_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(script_dir)
# sys.path.append(parent_dir)

sys.path.append("/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts")
import preprocess as preproc


#%% Test
import importlib
importlib.reload(preproc)

#config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts/testing/config_test_BS.yaml" # CHANGE if testing
#config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts/testing/config_test_STS.yaml" # CHANGE if testing
#config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts/testing/regression_testing/config_BS_reg_test.yml" # CHANGE if testing
config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/config/config_STS_Q.yml"


with open(config_path, 'r') as file:  # open config file
    config = yaml.safe_load(file)
    
cfg_dataset = config['dataset']
cfg_preprocess = config['preprocess']
cfg_hrf = config['hrf_estimation']
mse_amp_thresh = config['groupaverage']['mse']['mse_amp_thresh']

subjects = cfg_dataset['subject'] 
tasks = cfg_dataset['task'] 
runs = cfg_dataset['run'] 

# Loop through lists of tasks, subjects, and runs
for subj in subjects:
    for task in tasks:
        for run in runs:
            cfg_preprocess_loop = copy.deepcopy(cfg_preprocess)
            cfg_hrf_loop = copy.deepcopy(cfg_hrf)
            mse_amp_thresh_loop = copy.deepcopy(mse_amp_thresh)
            
            snirf_path = f"{cfg_dataset['root_dir']}/sub-{subj}/nirs/sub-{subj}_task-{task}_run-{run}_nirs.snirf"
            events_path =  f"{cfg_dataset['root_dir']}/sub-{subj}/nirs/sub-{subj}_task-{task}_run-{run}_events.tsv"
            
            save_path = f"{cfg_dataset['root_dir']}/derivatives/{cfg_dataset['derivatives_subfolder']}/preprocessed_data/sub-{subj}/"
            
            #out_snirf = f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_preprocessed.snirf"
            #out_json = f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_dataquality.json"
            
            out_files = {
                "out_snirf" : f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_preprocessed.pkl",
                "out_sidecar": f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_dataquality_geo.sidecar",
                #"out_dqr": f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_DQR.png",
                #"out_gvtd": f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_DQR_gvtd_hist.png",
                #"out_slope": f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_slope.png"
                }
            
            os.makedirs(save_path, exist_ok=True)  # make directory if it doesn't already exist

            print(f"Processing sub-{subj}, task-{task}, run-{run}...")
            
            preproc.preprocess_func(config, snirf_path, events_path, cfg_dataset, cfg_preprocess_loop, cfg_hrf_loop, mse_amp_thresh_loop, out_files)
            
