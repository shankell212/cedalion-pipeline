#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 09:41:22 2025

@author: smkelley
"""

import os
import gzip
import pickle
import tkinter as tk
from tkinter import filedialog

import cedalion
import xarray as xr
from cedalion import io, nirs, units
import cedalion.vis.plot_probe as vPlotProbe

#%%

#path2results = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS/derivatives/cedalion/test/"
path2results = "/projectnb/nphfnirs/s/datasets/Interactive_Walking_HD/derivatives/cedalion"
  
    
task = "STS"
sub = "01"  # only for opening rec
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
    geo2d = groupavg_results['geo2d']
    geo3d = groupavg_results['geo3d']


    print("Blockaverage file loaded successfully!")

else:
    print(f"Error: File '{filepath_bl}' not found!")
    
# groupaverage = groupaverage_unweighted
    
#%% Open rec to get 2d adn 3d probe coords
    
# # open rec bc i dont have geo2d or 3d
# results_rec = os.path.join(path2results, "preprocessed_data/")
# filname_rec = "sub-" + sub + "/sub-" + sub + "_task-" + task + "_run-01_nirs_preprocessed.snirf"
# filepath_rec = os.path.join(results_rec , filname_rec)


# records = cedalion.io.read_snirf( filepath_rec )
# rec = records[0]

# geo3d = rec.geo3d
# geo2d = rec.geo2d


#%% Loop through all subjects


for idx, subj in enumerate(blockaverage_subj.subj.values):
    subj_data = blockaverage_subj.sel(subj=subj)
    groupaverage = subj_data

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
        
        # # stderr
        # blockaverage_stderr = blockaverage_stderr.rename({'reltime':'time'})
        # blockaverage_stderr.time.attrs['units'] = units.s
        # blockaverage_stderr_conc = nirs.od2conc(blockaverage_stderr, geo3d, dpf, spectrum="prahl")
        # blockaverage_stderr_conc = blockaverage_stderr_conc.rename({'time':'reltime'})
        
        # tstat_conc = groupaverage_conc / blockaverage_stderr_conc
        
        groupaverage_conc_plot = groupaverage_conc.transpose("trial_type", "channel", "chromo", "reltime")

        file_name = f"sub-{subj}_plot_probe.png"
        save_path = os.path.join(path2results, "plots", "plot_probe", "subjects") 
        os.makedirs(save_path, exist_ok=True)  # make directory if it doesn't already exist
        out_file = os.path.join(save_path, file_name)
        
        vPlotProbe.save_plot_probe_image( groupaverage_conc_plot,
            geo2d,
            geo3d,
            out_file = out_file,
            xscale = 0.5,
            yscale = 2.0,
            title = f'Subject {subj}',
            show_optode_labels=True,
            show_meas_lines=False,
        )
        
        print(f'Successfully saved plot probe for subject {subj} \n')
        
   
#%% Plot

#tstat = blockaverage / blockaverage_stderr   # tstat = blockavg/ noise

#vPlotProbe.run_vis(blockaverage = groupaverage_conc, geo2d = geo2d, geo3d = geo3d)








