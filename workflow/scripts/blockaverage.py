# -*- coding: utf-8 -*-
"""
Perform blockaverage to get HRF

Created on Thu Jun  5 09:40:42 2025

@author: shank
"""

import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality

import cedalion.models.glm as glm
import cedalion.plots as plots

from cedalion import units
import numpy as np
import xarray as xr

import matplotlib.pyplot as p
import yaml

import pdb


#%%
config = snakemake.config

# config_path = "C:\\Users\\shank\\Documents\\GitHub\\cedalion-pipeline\\workflow\\config\\config.yaml"

# with open(config_path, 'r') as file:
#     config = yaml.safe_load(file)

cfg_dataset = config["dataset"]
cfg_hrf = config["hrf"]
cfg_blockaverage = config["blockaverage"]


if not cfg_dataset['derivatives_subfolder']:
    cfg_dataset['derivatives_subfolder'] = ''

run_files = snakemake.input.preproc_runs

#run_files = ["/myproj/sub-01/nirs/sub-01_task-nback_run-1_nirs_preprocessed.snirf"]

records = cedalion.io.read_snirf( run_files ) 

rec = records[0]

save_path = snakemake.output

print(f'Run files: \n {run_files}')
print(f'Record: \n {records}')
print(f'rec: \n {rec}')



#%%

#%% 
# Check if rec_str exists for current subject
# if cfg_blockaverage['rec_str'] not in rec.timeseries:
#     print(f"{cfg_blockaverage['rec_str']} does not exist for subject {snakemake.wildcards.subject}, task {snakemake.wildcards.task}, \
#           run {snakemake.wildcards.run}.  Skipping this file.")
#     continue  # if rec_str does not exist, skip 
# else:
#     ts = rec[cfg_blockaverage['rec_str']].copy()     # !!! make this a try except statement 
    
# ts = rec[cfg_blockaverage['rec_str']].copy()

# # select the stim for the given file
# stim = rec.stim.copy()
    
# # get the epochs
# # check if ts has dimenstion chromo
# if 'chromo' in ts.dims:
#     ts = ts.transpose('chromo', 'channel', 'time')
# else:
#     ts = ts.transpose('wavelength', 'channel', 'time')
# ts = ts.assign_coords(samples=('time', np.arange(len(ts.time))))
# ts['time'] = ts.time.pint.quantify(units.s)     

# #
# # block average
# #
# epochs_tmp = ts.cd.to_epochs(
#                             stim,  # stimulus dataframe
#                             set(stim[stim.trial_type.isin(cfg_blockaverage['hrf']['stim_lst'])].trial_type), # select events
#                             before = cfg_blockaverage['hrf']['t_pre'],  # seconds before stimulus
#                             after = cfg_blockaverage['hrf']['t_post'],  # seconds after stimulus
#                         )


# baseline = epochs_all.sel(reltime=(epochs_all.reltime < 0)).mean('reltime')
# epochs = epochs_all - baseline
# blockaverage = epochs.groupby('trial_type').mean('epoch') # mean across all epochs



