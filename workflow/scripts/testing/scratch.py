#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 17:44:18 2025

@author: smkelley
"""
import cedalion
import os
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt 
import yaml

#%%

root_data_dir = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS/derivatives/processed_data/"
filepath = os.path.join(root_data_dir, "od_o_postglm_ind_blockaverage_for_IR.pkl.gz")

rec_str = "od_o_postglm"

print("Loading saved data")
with gzip.open( os.path.join(root_data_dir, rec_str+'_ind_blockaverage_for_IR.pkl.gz'), 'rb') as f:
     all_results = pickle.load(f)
     
blockaverage = all_results['blockaverage']  # group_blockaverage unweighted rename
blockaverage_weighted = all_results['blockaverage_weighted']
blockaverage_stderr = all_results['blockaverage_stderr']
blockaverage_subj = all_results['blockaverage_subj'] # always unweighted   - load into img recon
blockaverage_mse_subj = all_results['blockaverage_mse_subj'] # - load into img recon
geo2d = all_results['geo2d']
geo3d = all_results['geo3d']



# Plot mse_hist
mse_subj_stacked = blockaverage_mse_subj.stack(foo=['subj', 'trial_type', 'channel', 'wavelength', 'reltime'])

f, ax = plt.subplots()
ax.hist(np.log10(mse_subj_stacked), bins=100)


source = 'S55'
matching_channels = [ch for ch in blockaverage_weighted.channel.values if source in ch]

TRIAL = 'right'
CHANNEL = matching_channels[8] #'S5D137'
WAV = 850  #760
grp_right = blockaverage_weighted.sel(wavelength=WAV, channel=CHANNEL, trial_type='right')
grp_left = blockaverage_weighted.sel(wavelength=WAV, channel=CHANNEL, trial_type='left')

f, ax = plt.subplots(1,1)
ax.plot(grp_right.reltime, grp_right, label='right')
ax.plot(grp_left.reltime, grp_left, label='left')

plt.ylabel('OD')
plt.xlabel('time (s)')
plt.legend()
plt.title(f'group avg weighted (wav {WAV}, channel {CHANNEL}')
        

           