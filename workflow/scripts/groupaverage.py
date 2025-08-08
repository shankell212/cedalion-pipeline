#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 12:55:08 2025

@author: smkelley
"""
import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality

import cedalion.models.glm as glm
import cedalion.plots as plots

from cedalion.physunits import units
import pint
import numpy as np
import xarray as xr

import matplotlib.pyplot as p

import yaml
import gzip
import pickle
import json
import pdb


#%%
def groupaverage_func(cfg_dataset, cfg_blockaverage, cfg_hrf, cfg_groupaverage, flag_prune_channels, blockavg_files, data_quality_files, geo_files, out):
    print("group averaging: \n")
    print(blockavg_files)
    
    # Loop thru trial tpes
    for idxt, trial_type in enumerate(cfg_hrf['stim_lst']): 
        
        # Loop over subjects
        blockaverage_subj = None
        for subj_idx, subj in enumerate(blockavg_files):
    
            # Load in current sub's blockaverage file
            with open(subj, 'rb') as f:
                results = pickle.load(f)
            blockaverage = results['blockaverage']
            mse_t = results['mse_t']
        
            blockaverage1 = blockaverage.sel(trial_type=trial_type)  # select current trial type
            blockaverage1 = blockaverage1.expand_dims('trial_type')  # readd trial type coord
            blockaverage = blockaverage1.copy()
            
            mse_t1 = mse_t.sel(trial_type=trial_type)  # select current trial type
            mse_t1 = mse_t1.expand_dims('trial_type')  # readd trial type coord
            mse_t = mse_t1.copy()
            
            blockaverage_weighted = blockaverage.copy()
            
            # gather the blockaverage across subjects
            if blockaverage_subj is None:
                                
                blockaverage_subj_weighted = blockaverage_weighted / mse_t 

                blockaverage_mean_weighted = blockaverage_weighted / mse_t
                sum_mse_inv = 1/mse_t
                
                # add a subject dimension and coordinate
                blockaverage_subj = blockaverage.expand_dims('subj')
                blockaverage_subj = blockaverage_subj.assign_coords(subj=[cfg_dataset['subject'][subj_idx]])
     
                mse_subj = mse_t.expand_dims('subj') 
                mse_subj = mse_subj.assign_coords(subj=[cfg_dataset['subject'][subj_idx]])
                
            else:
                                
                blockaverage_subj_weighted = xr.concat([blockaverage_subj_weighted, blockaverage_weighted / mse_t], dim='subj')
                
                blockaverage_mean_weighted = blockaverage_mean_weighted + blockaverage_weighted  / mse_t
                sum_mse_inv = sum_mse_inv + 1 / mse_t  
                
                blockaverage_tmp = blockaverage.expand_dims('subj')
                blockaverage_tmp = blockaverage_tmp.assign_coords(subj=[cfg_dataset['subject'][subj_idx]])

                blockaverage_subj = xr.concat([blockaverage_subj, blockaverage_tmp], dim='subj')
    
                mse_subj_tmp = mse_t.expand_dims('subj')
                mse_subj_tmp = mse_subj_tmp.assign_coords(subj=[cfg_dataset['subject'][subj_idx]])

                mse_subj = xr.concat([mse_subj, mse_subj_tmp], dim='subj')
    
            
        # DONE LOOP OVER SUBJECTS
            
#%%
def main():
    config = snakemake.config
    
    cfg_dataset = snakemake.params.cfg_dataset  # get params
    cfg_blockaverage = snakemake.params.cfg_blockaverage
    cfg_hrf = snakemake.params.cfg_hrf
    cfg_groupaverage = snakemake.params.cfg_groupaverage
    cfg_groupaverage['mse_amp_thresh'] = snakemake.params.mse_amp_thresh
    flag_prune_channels = snakemake.params.flag_prune_channels
    
    blockavg_files = snakemake.input.blockavg_subs  #.preproc_runs
    data_quality_files = snakemake.input.quality
    geo_files = snakemake.input.geo
    #blockavg_files_nc = snakemake.input.blockavg_nc
    #epoch_files_nc = snakemake.input.epochs_nc
    
    out = snakemake.output[0]

    groupaverage_func(cfg_dataset, cfg_blockaverage, cfg_hrf, cfg_groupaverage, flag_prune_channels, blockavg_files, data_quality_files, geo_files, out)
    
            
if __name__ == "__main__":
    main()
