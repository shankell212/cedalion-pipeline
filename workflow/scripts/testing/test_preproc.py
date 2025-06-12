#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 07:23:40 2025

@author: smkelley
"""

import yaml
import os
import sys
sys.path.append('/projectnb/nphfnirs/ns/Shannon/Code/cedalion-pipeline/workflow/scripts/')
import preprocess as preproc



config_path = "/projectnb/nphfnirs/ns/Shannon/Code/cedalion-pipeline/workflow/config/config.yaml" # CHANGE if testing
#config_path = "C:\\Users\\shank\\Documents\\GitHub\\cedalion-pipeline\\workflow\\config\\config.yaml"

with open(config_path, 'r') as file:  # open config file
    config = yaml.safe_load(file)
    
cfg_dataset = config['dataset']
cfg_preprocess = config['preprocess']

subj = cfg_dataset['subject'][0]   # sub idx you want to test
task = cfg_dataset['task'][0]
run = cfg_dataset['run'][0]

snirf_path = f"{cfg_dataset['root_dir']}/sub-{subj}/nirs/sub-{subj}_task-{task}_run-{run}_nirs.snirf"
events_path =  f"{cfg_dataset['root_dir']}/sub-{subj}/nirs/sub-{subj}_task-{task}_run-{run}_events.tsv"

save_path = f"{cfg_dataset['root_dir']}/derivatives/{cfg_dataset['derivatives_subfolder']}/preprocessed_data/sub-{subj}/"
out_snirf = f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_preprocessed.snirf"
out_json = f"{save_path}sub-{subj}_task-{task}_run-{run}_nirs_dataquality.json"

der_dir = os.path.join(save_path)
if not os.path.exists(der_dir):
    os.makedirs(der_dir)


preproc.preprocess_func(config, snirf_path, events_path, cfg_dataset, cfg_preprocess, out_snirf, out_json)