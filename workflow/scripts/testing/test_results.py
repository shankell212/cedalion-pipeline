#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 12:00:17 2025

@author: smkelley
"""

#%% Imports
import cedalion
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
import pickle
import gzip


#%%

# TEST that saved results look good

config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts/testing/config_test_BS.yaml" # CHANGE if testing
# config_path = "/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/config/config.yaml"

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

cfg_dataset = config['dataset']
subject = cfg_dataset['subject']
task = cfg_dataset['task']
run = cfg_dataset['run']


#%% TEST preproc
preproc_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'preprocessed_data')
preproc_files = [[[os.path.join(f'sub-{s}',f'sub-{s}_task-{t}_run-{r}_nirs_preprocessed.pkl') for r in run] for t in task] for s in subject]
dataqual_files = [[[os.path.join(f'sub-{s}',f'sub-{s}_task-{t}_run-{r}_nirs_dataquality_geo.sidecar') for r in run] for t in task] for s in subject]

# grab 1 subject file
subj_idx = 0 # CHANGE
task_idx = 0
run_idx = 0
preproc_singlefile = preproc_files[subj_idx][task_idx][run_idx]
dataqual_singlefile = dataqual_files[subj_idx][task_idx][run_idx]  

with gzip.open(os.path.join(preproc_dir, preproc_singlefile), 'rb') as f:  # open pkl
    record = pickle.load(f)
    rec = record[0]
    
print(rec)
    
with gzip.open(os.path.join(preproc_dir, dataqual_singlefile), 'rb') as f:  # open dataqual sidecar
    data_qual = pickle.load(f)
    
# get good chans
chs_pruned = data_qual['chs_pruned']  # good chans = 0.58

mask = chs_pruned.values == 0.58
good_chans = chs_pruned.channel.values[mask].tolist()

od = rec['od']
od_corr = rec['od_corrected']

CHANNEL = good_chans[60]  # CHANGE
CAHNNEL='S5D137'
WAV_IDX = 1

od_ch = od.sel(channel=CHANNEL, wavelength=od.wavelength[WAV_IDX])
od_corr_ch = od_corr.sel(channel=CHANNEL, wavelength=od_corr.wavelength[WAV_IDX])


# plot od and od_corrected
f, ax = plt.subplots(1,1)
ax.plot(od_ch.time, od_ch, 'b', label='od')
ax.plot(od_corr_ch.time, od_corr_ch, 'g', label='od_corrected')

plt.xlabel('time (s)')
plt.ylabel('OD')
plt.title(f'Comparing OD before and after correction - channel {CHANNEL}')
plt.legend()

# # plot only od
# f, ax = plt.subplots(1,1)
# plt.plot(od_ch.time, od_ch, 'b', label='od')
# plt.xlabel('time (s)')
# plt.ylabel('OD')
# plt.title('OD before correction')
# plt.legend()

# # plot only od
# f, ax = plt.subplots(1,1)
# plt.plot(od_corr_ch.time, od_corr_ch, 'g', label='od_corrected')
# plt.xlabel('time (s)')
# plt.ylabel('OD')
# plt.title('OD after correction')
# plt.legend()

# Plot conc
conc_ch = rec['conc']

f, ax = plt.subplots(1,1)
ax.plot(conc_ch.time, conc_ch.sel(chromo='HbO', channel=CHANNEL), 'r', label='HbO')
ax.plot(conc_ch.time, conc_ch.sel(chromo='HbR', channel=CHANNEL), 'b', label='HbR')

plt.xlabel('time (s)')
plt.ylabel(str(rec["conc"].pint.units))
plt.title(f'Conc - channel {CHANNEL}')
plt.legend()

# Shannon/Cedalion == no GLM, commented out replacing mse_min, only grabbed last run data qual file
# Shannon/Cedalion/TEST == everything good, changed t post
#%% TEST groupaverage
groupavg_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'groupaverage')
groupavg_files = [f'task-{t}_nirs_groupaverage.pkl' for t in cfg_dataset['task']]


# Load in data
groupaverage_path = os.path.join(groupavg_dir, groupavg_files[0])
if os.path.exists(groupaverage_path):
    with open(groupaverage_path, 'rb') as f:
        groupavg_results = pickle.load(f)
  
    blockaverage_weighted = groupavg_results['group_blockaverage_weighted']  #groupavg_results['group_blockaverage_weighted']
    blockaverage = groupavg_results['group_blockaverage']
    blockaverage_stderr = groupavg_results['total_stderr_blockaverage']
    blockaverage_subj = groupavg_results['blockaverage_subj']
    blockaverage_mse_subj = groupavg_results['blockaverage_mse_subj']
    #geo2d = groupavg_results['geo2d']
    #geo3d = groupavg_results['geo3d']  # !!! this is not in groupaverage results yet
    print(f" {groupaverage_path} loaded successfully!")

else:
    print(f"Error: File '{groupaverage_path}' not found!")
    
SUBJ_IDX = 1
TRIAL_IDX = 1
WAV=1

# Plot mse_hist
mse_subj_stacked = blockaverage_mse_subj.stack(foo=['subj', 'trial_type', 'channel', 'wavelength', 'reltime'])

f, ax = plt.subplots()
ax.hist(np.log10(mse_subj_stacked), bins=100)

# find chans w this source
source = 'S55'
matching_channels = [ch for ch in blockaverage_weighted.channel.values if source in ch]

# Plot a chans from blockaverage
TRIAL = 'right'
CHANNEL = matching_channels[2] #'S5D137'
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
        

         

