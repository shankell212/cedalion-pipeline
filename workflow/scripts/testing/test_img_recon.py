#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 05:15:40 2025

@author: smkelley
"""

#%% Import modules

import yaml
import os
import sys
import copy
# script_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(script_dir)
# sys.path.append(parent_dir)
sys.path.append("/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts")
import image_recon as img_recon


#%% Load config and test function 
import importlib
importlib.reload(img_recon)
# 
# config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/config/config_IWHD_Q.yml"
# config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts/testing/config_test_BS.yaml"
#config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts/testing/regression_testing/config_BS_reg_test.yml" # CHANGE if testing
# config_path = "/projectnb/nphfnirs/s/datasets/Interactive_Walking_HD/derivatives/cedalion/new_inclQ_test_imgrecon_newnoise/config_STS_Q.yml"
config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/config/config.yaml"


with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

cfg_dataset = config['dataset']
cfg_img_recon = config['image_recon']
cfg_hrf = config['hrf_estimation']
task = cfg_dataset['task'][0]  # choose first task for testing
dirs = os.listdir(cfg_dataset['root_dir'])
subjects = [d.replace("sub-", "") for d in dirs if "sub" in d and d.replace("sub-", "") not in config["dataset"]["subjects_to_exclude"]]



Adot_path = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'cedalion', 'forward', config['image_recon']['generate_sensitivity']['sub_folder'], 'sensitivity.nc')
hrf_dir = os.path.join(cfg_dataset['root_dir'], "derivatives", "cedalion", cfg_dataset['derivatives_subfolder'], "hrf_estimate")  #, f"sub-{subj}")
blockavg_files = [os.path.join(hrf_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_hrf_estimate_{cfg_hrf['rec_str']}.nc") for subj in subjects ]

if cfg_img_recon['spatial_basis']['enable']:
    SB_path = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'cedalion', 'forward', cfg_img_recon['generate_sensitivity']['sub_folder'], 'sbf.nc')
else:
    SB_path= [] # Return an empty list if the input is not needed
#%
for idx, subj in enumerate(subjects):
    cfg_img_recon_loop = copy.deepcopy(cfg_img_recon)
    
    # create folders for each subj
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives',  'cedalion', cfg_dataset['derivatives_subfolder'], 'image_results', f"sub-{subj}")
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)

    file_name = (
        f"Xs_sub-{subj}_{task}"
        + f"_cov_alpha_spatial_{config['image_recon']['alpha_spatial']}"
        + f"_alpha_meas_{config['image_recon']['alpha_meas']}"
        + (f"_recon_mode_{config['image_recon']['recon_mode']}")
        + ("_Cmeas" if config["image_recon"]["Cmeas"]["enable"] else "_noCmeas")
        + ("_SB" if config["image_recon"]["spatial_basis"]["enable"] else "_noSB")
        + ("_mag" if config["image_recon"]["mag"]["enable"] else "_ts")
        + (f"_{config['image_recon']['mag']['t_win'][0]}_{config['image_recon']['mag']['t_win'][1]}" if config["image_recon"]["mag"]["enable"] else "")
        + ".nc"
    )

    out = os.path.join(der_dir, file_name)
#
    blockavg_file = blockavg_files[idx]

    img_recon.img_recon_func(cfg_img_recon_loop, cfg_hrf, blockavg_file, Adot_path, out, SB_path)



# %%
