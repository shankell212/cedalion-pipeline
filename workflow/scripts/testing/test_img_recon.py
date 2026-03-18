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
# config_path = "/projectnb/nphfnirs/s/datasets/Interactive_Walking_HD/derivatives/cedalion/new_inclQ_2_newnoise/config_STS_Q.yml"

root_dir = "/projectnb/nphfnirs/s/users/shannon/Data/test_data_cedalion_smk/data/"
config_path = os.path.join(root_dir, 'derivatives', 'cedalion', 'reference', 'ref_1', 'config_ref_1_tmp.yml')



with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

cfg_dataset = config['dataset']
cfg_img_recon = config['image_recon']
cfg_hrf = config['hrf_estimation']
task = cfg_dataset['task'][0]  # choose first task for testing

subjects = cfg_dataset['subject']
# subjects = ["20", "21", "22", "23", "24", "25", "26", "28"]

# # Get input file path input file
# groupaverage_dir = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "groupaverage")
# groupaverage_path = os.path.join(groupaverage_dir, f"task-{task}_nirs_groupaverage.pkl")
        
hrf_dir = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "hrf_estimate")  #, f"sub-{subj}")
blockavg_files = [os.path.join(hrf_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_hrf_estimate_{cfg_hrf['rec_str']}.pkl.gz") for subj in subjects ]
data_quality_files = [os.path.join(hrf_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_dataquality.json") for subj in subjects ]
geo_files = [os.path.join(hrf_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_geo.sidecar") for subj in subjects ]

#%
for idx, subj in enumerate(subjects):
    cfg_img_recon_loop = copy.deepcopy(cfg_img_recon)
    
    # create folders for each subj
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives',  cfg_dataset['derivatives_subfolder'], 'image_results', f"sub-{subj}")
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)

    file_name = (
        f"Xs_sub-{subj}_{task}"
        + f"_cov_alpha_spatial_{config['image_recon']['alpha_spatial']}"
        + f"_alpha_meas_{config['image_recon']['alpha_meas']}"
        + ("_direct" if config["image_recon"]["DIRECT"]["enable"] else "_indirect")
        + ("_Cmeas" if config["image_recon"]["Cmeas"]["enable"] else "_noCmeas")
        + ("_SB" if config["image_recon"]["spatial_basis"]["enable"] else "_noSB")
        + ("_mag" if config["image_recon"]["mag"]["enable"] else "_ts")
        + (f"_{config['image_recon']['mag']['t_win'][0]}_{config['image_recon']['mag']['t_win'][1]}" if config["image_recon"]["mag"]["enable"] else "")
        + ".pkl.gz"
    )

    out = os.path.join(der_dir, file_name)
#
    blockavg_file = blockavg_files[idx]

    img_recon.img_recon_func(cfg_dataset, cfg_img_recon_loop, cfg_hrf, blockavg_file, out)



# %%
