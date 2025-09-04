#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 06:12:33 2025

@author: smkelley
"""
#%% Imports

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


#%%
# path2results = "/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/derivatives/cedalion"

# path2results = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS/derivatives/Shannon/cedalion"
path2results = "/projectnb/nphfnirs/s/datasets/Interactive_Walking_HD/derivatives/cedalion"

task = "STS"

# sub = "547"  # only for opening rec
filname = "task-" + task + "_nirs_groupaverage.pkl"
filepath_bl = os.path.join(os.path.join(path2results, "groupaverage/") , filname)
    


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
    
    # stderr
    blockaverage_stderr = blockaverage_stderr.rename({'reltime':'time'})
    blockaverage_stderr.time.attrs['units'] = units.s
    blockaverage_stderr_conc = nirs.od2conc(blockaverage_stderr, geo3d, dpf, spectrum="prahl")
    blockaverage_stderr_conc = blockaverage_stderr_conc.rename({'time':'reltime'})
    
    tstat_conc = groupaverage_conc / blockaverage_stderr_conc
   
#%% Plot

#tstat = blockaverage / blockaverage_stderr   # tstat = blockavg/ noise

vPlotProbe.run_vis(blockaverage = groupaverage_conc, geo2d = geo2d, geo3d = geo3d)


