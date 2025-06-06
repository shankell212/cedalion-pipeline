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

import numpy as np
import xarray as xr
import pint
from cedalion.physunits import units

import matplotlib.pyplot as p
import yaml
import gzip
import pickle
import pdb


#%%

def blockaverage_func(cfg_dataset, cfg_blockaverage, cfg_hrf, run_files, out):
    
    # update units 
    cfg_hrf['t_pre']= units(cfg_hrf['t_pre'])
    cfg_hrf['t_post']= units(cfg_hrf['t_post'])
    
    # loop through files
    for file_idx, run in enumerate(run_files):
    
        # Check if rec_str exists for current subject
        # if cfg_blockaverage['rec_str'] not in rec.timeseries:
        #     print(f"{cfg_blockaverage['rec_str']} does not exist for subject {snakemake.wildcards.subject}, task {snakemake.wildcards.task}, \
        #           run {snakemake.wildcards.run}.  Skipping this file.")
        #     continue  # if rec_str does not exist, skip 
        # else:
        #     ts = rec[cfg_blockaverage['rec_str']].copy()     # !!! make this a try except statement 
            
        
        # if file path/ current run does not exist for this file, continue without it  (i.e. subj dropped out)
        if not os.path.isfile(run):
            continue   
        
        records = cedalion.io.read_snirf( run ) 
        rec = records[0]
        ts = rec[cfg_blockaverage['rec_str']].copy()
        stim = rec.stim.copy() # select the stim for the given file
            
        # check if ts has dimenstion chromo
        if 'chromo' in ts.dims:
            ts = ts.transpose('chromo', 'channel', 'time')
        else:
            ts = ts.transpose('wavelength', 'channel', 'time')
        ts = ts.assign_coords(samples=('time', np.arange(len(ts.time))))
        ts['time'] = ts.time.pint.quantify(units.s)     
    
        # get the epochs
        epochs_tmp = ts.cd.to_epochs(
                                    stim,  # stimulus dataframe
                                    set(stim[stim.trial_type.isin(cfg_hrf['stim_lst'])].trial_type), # select events
                                    before = cfg_hrf['t_pre'],  # seconds before stimulus
                                    after = cfg_hrf['t_post'],  # seconds after stimulus
                                )
        if file_idx == 0:
            epochs_all = epochs_tmp
        else:
            epochs_all = xr.concat([epochs_all, epochs_tmp], dim='epoch')  # concatinate epochs from all runs
    
        # DONE LOOP OVER FILES
    
    # Block Average
    baseline = epochs_all.sel(reltime=(epochs_all.reltime < 0)).mean('reltime')
    epochs = epochs_all - baseline  # baseline subtract
    blockaverage = epochs.groupby('trial_type').mean('epoch') # mean across all epochs


    # create new rec variable that only includes blockaverage for all rusn for this sub/task
    rec["blockaverage"] = blockaverage
    
    # remove all other keys except blockaverage timeseries
    for key in list(rec.timeseries.keys()):
        if key == "blockaverage":
            continue
        del rec.timeseries[key]
    
    rec.stim.duration = 1
    rec.stim.onset = 1
    rec.stim.value = 1
        
    for key in list(rec.aux_ts.keys()):
        del rec.aux_ts[key]
    
    # save data a pickle for now
    with open(out, "wb") as f:        # if output is a single string, it wraps it in an output object and need to index in
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Block average data saved successfully")
    
    # # Debugging issue with save snirf:
    # for key, timeseries in rec.timeseries.items():
    #     data_type = rec.get_timeseries_type(key)
    #     print(key)
    #     print(data_type)
    #     print(timeseries.dims)
        
    #     print('\n')
        
    #cedalion.io.snirf.write_snirf(out, rec)
    
    # PROCEED w/ saving as a pickle file for now
        # post prob on cedalion implementation
    
 
#%%

def main():
    try:

        config = snakemake.config
        
        cfg_dataset = snakemake.params.cfg_dataset
        cfg_blockaverage = snakemake.params.cfg_blockaverage
        cfg_hrf = snakemake.params.cfg_hrf
        run_files = snakemake.input  #.preproc_runs
        
        out = snakemake.output[0]
        
    except:
        config_path = "/projectnb/nphfnirs/ns/Shannon/Code/cedalion-pipeline/workflow/config/config.yaml"
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        cfg_dataset = config['dataset']
        cfg_blockaverage = config['blockaverage']
        cfg_hrf = config['hrf']
        
        subj = cfg_dataset['subject'][0]   # sub idx you want to test
        task = cfg_dataset['task'][0]
        run = cfg_dataset['run']
        
        preproc_dir = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "preprocessed_data")
        
        run_files = [os.path.join(preproc_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_run-{r}_nirs_preprocessed.snirf") for r in run]
        
        # run_files = ["/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/derivatives/cedalion/preprocessed_data/sub-01/sub-01_task-STS_run-01_nirs_preprocessed.snirf"]
        
        save_path = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "blockaverage", f"sub-{subj}")
        out = os.path.join(save_path,f"sub-{subj}_task-{task}_nirs_blockaverage.snirf")
        
        der_dir = os.path.join(save_path)
        if not os.path.exists(der_dir):
            os.makedirs(der_dir)
            
    blockaverage_func(cfg_dataset, cfg_blockaverage, cfg_hrf, run_files, out)

if __name__ == "__main__":
    main()
    
    


