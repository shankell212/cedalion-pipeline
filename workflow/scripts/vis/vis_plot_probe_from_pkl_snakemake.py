#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 06:12:33 2025

@author: smkelley
"""


import os
import gzip
import pickle
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# import cedalion
import xarray as xr
from cedalion import io, nirs, units
import cedalion.vis.plot_probe as vPlotProbe
import cedalion.geometry.registration as registration
import numpy as np

#%%
# path2results = "/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/derivatives/cedalion"

# path2results = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS/derivatives/Shannon/cedalion"
path2results = "/projectnb/nphfnirs/s/datasets/Interactive_Walking_HD/derivatives/cedalion"

task = "STS"

filname = "task-" + task + "_nirs_groupaverage.pkl"
filepath_bl = os.path.join(os.path.join(path2results, "groupaverage/") , filname)

  
# if os.path.exists(filepath_bl):
#     with gzip.open(filepath_bl, 'rb') as f:
#         groupavg_results = pickle.load(f)
#     blockaverage = groupavg_results['blockaverage']
#     blockaverage_stderr = groupavg_results['blockaverage_stderr']
#     blockaverage_subj = groupavg_results['blockaverage_subj']
#     blockaverage_mse_subj = groupavg_results['blockaverage_mse_subj']
#     geo2d = groupavg_results['geo2d']
#     geo3d = groupavg_results['geo3d']
#     print("Blockaverage file loaded successfully!")

# else:
#     print(f"Error: File '{filepath_bl}' not found!")
    


if os.path.exists(filepath_bl):
    with open(filepath_bl, 'rb') as f:
        groupavg_results = pickle.load(f)
        
    groupaverage = groupavg_results['group_blockaverage_weighted']
    groupaverage_unweighted = groupavg_results['group_blockaverage']
    blockaverage_stderr = groupavg_results['total_stderr_blockaverage']
    blockaverage_subj = groupavg_results['blockaverage_subj']
    blockaverage_mse_subj = groupavg_results['blockaverage_mse_subj']
    #geo2d = groupavg_results['geo2d']
    geo3d = groupavg_results['geo3d']
    
    # groupaverage_unweighted_orig = groupavg_results['group_blockaverage_orig']
    # blockaverage_subj_orig = groupavg_results['blockaverage_subj_orig']
    print("Blockaverage file loaded successfully!")

else:
    raise ValueError(f"Error: File '{filepath_bl}' not found!")
    
# groupaverage = groupaverage_unweighted_orig  # plot unweighted group averagegroupaverage_conc.sel(channel=ch_name)
geo2d = registration.simple_scalp_projection(geo3d)


#%% Convert to conc if in OD

# If blockaverage in OD, convert to conc

if 'wavelength' in groupaverage.dims:
    dpf = xr.DataArray(
        [1, 1],
        dims="wavelength",
       coords={"wavelength": groupaverage.wavelength},
    )

    # blockavg
    groupaverage = groupaverage.rename({'reltime':'time'})
    groupaverage.time.attrs['units'] = units.s
    groupaverage_conc = nirs.od2conc(groupaverage, geo3d, dpf, spectrum="prahl")
    groupaverage_conc = groupaverage_conc.rename({'time':'reltime'})
    
    blockaverage_subj = blockaverage_subj.rename({'reltime':'time'})
    blockaverage_subj.time.attrs['units'] = units.s
    blockaverage_subj_conc = nirs.od2conc(blockaverage_subj, geo3d, dpf, spectrum="prahl")
    blockaverage_subj_conc = blockaverage_subj_conc.rename({'time':'reltime'})
    
    # stderr
    blockaverage_stderr = blockaverage_stderr.rename({'reltime':'time'})
    blockaverage_stderr.time.attrs['units'] = units.s
    blockaverage_stderr_conc = nirs.od2conc(blockaverage_stderr, geo3d, dpf, spectrum="prahl")
    blockaverage_stderr_conc = blockaverage_stderr_conc.rename({'time':'reltime'})
    
    tstat_conc = groupaverage_conc / blockaverage_stderr_conc
   
#%% Plot

#tstat = blockaverage / blockaverage_stderr   # tstat = blockavg/ noise

# blk_avg = blockaverage_subj_conc.sel(subj='01')

# thresh = np.abs(groupaverage_conc).max() #.item() # if chan is => thresh make NaN
# thresh=thresh.values.item()
# thresh = float(thresh)
thresh = 100.0
exceeds = (np.abs(groupaverage_conc) >= thresh*units.micromolar).any(dim="reltime")

chan_indices = np.where(exceeds.values)[2]
groupavg_conc_new = groupaverage_conc.copy()
groupavg_conc_new[:,:,chan_indices,:] = np.nan*units.micromolar




#%%



vPlotProbe.run_vis(blockaverage = groupavg_conc_new, geo2d = geo2d, geo3d = geo3d)



#%% Plot single channel HRF

# chan = "S45D121"
# chan = "S47D29"
# chan = "S55D105"
# chan = "S53D24"
# chan = "S55D23"
# chan = "S53D32"
# chan = "S49D32"
# chan = "S53D23" # not bad

# channels = [
#     "S45D121", "S47D29", "S55D105", "S53D24", 
#     "S55D23", "S53D32", "S49D32", "S53D23"
# ]

# data_chan = groupaverage_conc.sel(channel=chan)

# color_map = {"HbO": "red", "HbR": "blue"}

# # Plot both HbO and HbR
# #plt.figure(figsize= (12,6)
# fig, axes = plt.subplots(nrows=4, ncols=2, figsize= (12,6))  #, sharex=True, sharey=True) #(10, 5))

# axes = axes.flatten()
# for i,chan in enumerate(channels):
#     data_chan = groupaverage_conc.sel(channel=chan)
chan = "S45D121"
chan = "S47D29"
chan = "S55D105"
chan = "S53D24"
chan = "S55D23"
chan = "S53D32"
chan = "S49D32"
chan = "S53D23" # not bad

channels = [
    "S45D121", "S47D29", "S55D105", "S53D24", 
    "S55D23", "S53D32", "S49D32", "S53D23"
]

data_chan = groupaverage_conc.sel(channel=chan)

color_map = {"HbO": "red", "HbR": "blue"}

# Plot both HbO and HbR
#plt.figure(figsize= (12,6)
fig, axes = plt.subplots(nrows=4, ncols=2, figsize= (12,6))  #, sharex=True, sharey=True) #(10, 5))

axes = axes.flatten()
for i,chan in enumerate(channels):
    data_chan = groupaverage_conc.sel(channel=chan)
    ax = axes[i]
    for chromo in data_chan.chromo.values:
        ax.plot(
            data_chan.reltime,
            data_chan.sel(chromo=chromo).squeeze(),
            label=chromo, 
            color = color_map.get(chromo)
        )
    ax.set_title(f"Channel {chan}", fontsize=10)
    ax.grid(True)

plt.title(f"Channel {chan}")
plt.xlabel("Time (s)")
plt.ylabel("Concentration (¼M)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()






#     ax = axes[i]
#     for chromo in data_chan.chromo.values:
#         ax.plot(
#             data_chan.reltime,
#             data_chan.sel(chromo=chromo).squeeze(),
#             label=chromo, 
#             color = color_map.get(chromo)
#         )
#     ax.set_title(f"Channel {chan}", fontsize=10)
#     ax.grid(True)

# plt.title(f"Channel {chan}")
# plt.xlabel("Time (s)")
# plt.ylabel("Concentration (¼M)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()





