#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 07:39:52 2025

@author: smkelley
"""

#%% Imports
import yaml
import os
import sys
# script_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(script_dir)
# sys.path.append(parent_dir)
sys.path.append("/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts")

import groupaverage as groupavg

#%% Run func

import importlib
importlib.reload(groupavg)

config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/config/config_STS.yaml"
#config_path = "C:\\Users\\shank\\Documents\\GitHub\\cedalion-pipeline\\workflow\\config\\config.yaml"
#config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts/testing/config_test_BS.yaml" # CHANGE if testing
#config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts/testing/regression_testing/config_BS_reg_test.yml" # CHANGE if testing

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# CHANGE
blockavg = False  # if true, group average done on blockavg, false, done on img recon

cfg_dataset = config['dataset']
# cfg_blockaverage = config['blockaverage']
cfg_hrf = config['hrf_estimation']
cfg_groupaverage = config['groupaverage']
# flag_prune_channels = config['preprocess']['steps']['prune']['enable']
# cfg_groupaverage['mse_amp_thresh'] = config['preprocess']['steps']['prune']['amp_thresh']


subjects = cfg_dataset['subject']
task = cfg_dataset['task'][0]

blockavg_dir = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "hrf_estimate")  #, f"sub-{subj}")
if blockavg:
    blockavg_files = [os.path.join(blockavg_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_hrf_estimate_{cfg_hrf['rec_str']}.pkl") for subj in subjects ]
else:
    file_name = (
        #f"Xs_sub-{subj}_{task}"
        f"_cov_alpha_spatial_{config['image_recon']['alpha_spatial']}"
        + f"_alpha_meas_{config['image_recon']['alpha_meas']}"
        + ("_direct" if config["image_recon"]["DIRECT"]["enable"] else "_indirect")
        + ("_Cmeas" if config["image_recon"]["Cmeas"]["enable"] else "_noCmeas")
        + ("_SB" if config["image_recon"]["spatial_basis"]["enable"] else "_noSB")
        + ("_mag" if config["image_recon"]["mag"]["enable"] else "_ts")
        + ".pkl.gz"
    )
    image_dir = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "image_results")
    image_files =  [os.path.join(image_dir, f"sub-{subj}", (f"Xs_sub-{subj}_{task}" + file_name)) for subj in subjects ]


data_quality_files = [os.path.join(blockavg_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_dataquality.json") for subj in subjects ]
geo_files = [os.path.join(blockavg_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_geo.sidecar") for subj in subjects ]

# blockavg_files_nc = [os.path.join(blockavg_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_blockaverage.nc") for subj in subjects ]
# epoch_files_nc = [os.path.join(blockavg_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_epochs.nc") for subj in subjects ]

# preproc_dir = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "preprocessed_data")  #, f"sub-{subj}")
# data_quality_files = [os.path.join(preproc_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_run-{cfg_dataset['run'][-1]}_nirs_dataquality_geo.sidecar") for subj in subjects ]


#save_path = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "groupaverage")

if blockavg:
    save_path = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "groupaverage")
    out = os.path.join(save_path, f"task-{task}_nirs_groupaverage_chanspace_{cfg_hrf['rec_str']}.pkl")
else:
    save_path = os.path.join(cfg_dataset['root_dir'], 'derivatives',  cfg_dataset['derivatives_subfolder'], 'image_results')
    file_name = (
        f"X_{task}_groupavg_"
        + f"_cov_alpha_spatial_{config['image_recon']['alpha_spatial']}"
        + f"_alpha_meas_{config['image_recon']['alpha_meas']}"
        + ("_direct" if config["image_recon"]["DIRECT"]["enable"] else "_indirect")
        + ("_Cmeas" if config["image_recon"]["Cmeas"]["enable"] else "_noCmeas")
        + ("_SB" if config["image_recon"]["spatial_basis"]["enable"] else "_noSB")
        + ("_mag" if config["image_recon"]["mag"]["enable"] else "_ts")
        + ".pkl.gz"
    )
    out = os.path.join(save_path, file_name)
der_dir = os.path.join(save_path)
if not os.path.exists(der_dir):
    os.makedirs(der_dir)

if blockavg:    
    groupavg.groupaverage_func(cfg_dataset, cfg_groupaverage, cfg_hrf, blockavg_files, geo_files, out)
else:
    groupavg.groupaverage_func(cfg_dataset, cfg_groupaverage, cfg_hrf, image_files, geo_files, out)

