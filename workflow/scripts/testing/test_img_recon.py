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
sys.path.append('/projectnb/nphfnirs/ns/Shannon/Code/cedalion-pipeline/workflow/scripts/')  # CHANGE
import image_recon as img_recon


#%% Load config and test function 
import importlib
importlib.reload(img_recon)

#config_path = "/projectnb/nphfnirs/ns/Shannon/Code/cedalion-pipeline/workflow/config/config.yaml"
config_path = "/projectnb/nphfnirs/ns/Shannon/Code/cedalion-pipeline/workflow/scripts/testing/config_test.yaml"


with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

cfg_dataset = config['dataset']
cfg_img_recon = config['image_recon']
task = cfg_dataset['task'][0]  # choose first task for testing



# Get input file path input file
groupaverage_dir = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "groupaverage")
groupaverage_path = os.path.join(groupaverage_dir, f"task-{task}_nirs_groupaverage.pkl")



# if cfg_img_recon['DIRECT']['enable']:
#     direct_name = 'direct'
# else:
#     direct_name = 'indirect'
    
# if cfg_img_recon['SB']['enable']:
#     SB_name = 'SB'
# else:
#     SB_name = 'noSB'

# if cfg_img_recon['Cmeas']['enable']:
#     Cmeas_name = 'Cmeas'
# else:
#     Cmeas_name = 'noCmeas'
    
der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives',  cfg_dataset['derivatives_subfolder'], 'image_results')
if not os.path.exists(der_dir):
    os.makedirs(der_dir)

        
out = (
    f"Xs_{task}"
    + f"_cov_alpha_spatial_{config['image_recon']['alpha_spatial']}"
    + f"_alpha_meas_{config['image_recon']['alpha_meas']}"
    + ("_direct" if config["image_recon"]["DIRECT"]["enable"] else "_indirect")
    + ("_Cmeas" if config["image_recon"]["Cmeas"]["enable"] else "_noCmeas")
    + ("_SB" if config["image_recon"]["spatial_basis"]["enable"] else "_noSB")
    + ".pkl.gz"
)


img_recon.img_recon_func(cfg_dataset, cfg_img_recon, groupaverage_path, out)


