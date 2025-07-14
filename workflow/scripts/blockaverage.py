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
import json
import pdb


#%%

def blockaverage_func(cfg_dataset, cfg_blockaverage, cfg_hrf, run_files, data_quality_files, out_pkl, out_json, out_geo):  #, out_blkavg_nc, out_epoch_nc):
    print(f'run_files: {run_files}')
    print(f'data_quality_files: {data_quality_files}')
    # update units 
    cfg_hrf['t_pre']= units(cfg_hrf['t_pre'])
    cfg_hrf['t_post']= units(cfg_hrf['t_post'])
    
    # Loop through files
    idx_sat_runs = []
    idx_amp_runs = []
    bad_chans_sat_runs = []
    bad_chans_amp_runs = []
    
    for file_idx, run in enumerate(run_files):
    
        # Check if rec_str exists for current subject
        # if cfg_blockaverage['rec_str'] not in rec.timeseries:
        #     print(f"{cfg_blockaverage['rec_str']} does not exist for subject {snakemake.wildcards.subject}, task {snakemake.wildcards.task}, \
        #           run {snakemake.wildcards.run}.  Skipping this file.")
        #     continue  # if rec_str does not exist, skip 
        # else:
        #     ts = rec[cfg_blockaverage['rec_str']].copy()     # !!! make this a try except statement 
            
        
        # if file path/ current run does not exist for this file, continue without it  (i.e. subj dropped out)
        # if not os.path.isfile(run):  # !!! do not need tis check anymore?
        #     continue   
        
        # # Load in snirf for curr subj and run
        # records = cedalion.io.read_snirf( run ) 
        # rec = records[0]
        with gzip.open(run, 'rb') as f:
            record = pickle.load(f)
            rec = record[0]
        ts = rec[cfg_blockaverage['rec_str']].copy()
        stim = rec.stim.copy() # select the stim for the given file
        
        # Load in json data qual
        # with open(data_quality_files[file_idx], 'r') as fp:
        #     data_quality_run = json.load(fp)
        with gzip.open(data_quality_files[file_idx], 'rb') as f:
            data_quality_run = pickle.load(f)
            
        geo2d = data_quality_run['geo2d']
        geo3d = data_quality_run['geo3d']
            
        # check if ts has dimenstion chromo
        if 'chromo' in ts.dims:
            ts = ts.transpose('chromo', 'channel', 'time')  # !!! try transpose(..., 'channel', 'time') to get rid of if statement
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
            epochs_all = xr.concat([epochs_all, epochs_tmp], dim='epoch')  # concatenate epochs from all runs
    
        # Concatenate all data data qual stuff
        idx_sat_runs.append(data_quality_run['idx_sat'])
        bad_chans_sat_runs.append(data_quality_run['bad_chans_sat'])
        idx_amp_runs.append(data_quality_run['idx_amp'])
        bad_chans_amp_runs.append(data_quality_run['bad_chans_amp'])
        
        # DONE LOOP OVER FILES

    # Block Average
    baseline = epochs_all.sel(reltime=(epochs_all.reltime < 0)).mean('reltime')
    epochs = epochs_all - baseline  # baseline subtract
    blockaverage = epochs.groupby('trial_type').mean('epoch') # mean across all epochs 
    
    # # create new rec variable that only includes blockaverage for all rusn for this sub/task
    # rec["blockaverage"] = blockaverage
    # rec['epochs'] = epochs
    
    # # remove all other keys except blockaverage timeseries
    # for key in list(rec.timeseries.keys()):
    #     if key == "blockaverage" or key == "epochs":
    #         continue
    #     del rec.timeseries[key]
    
    # rec.stim.duration = 1
    # rec.stim.onset = 1
    # rec.stim.value = 1
        
    # for key in list(rec.aux_ts.keys()):
    #     del rec.aux_ts[key]
    
    # Save geometric 2d and 3d positions to sidecar file
    geo_sidecar = {
        'geo2d': geo2d,
        'geo3d': geo3d
        }
    file = gzip.GzipFile(out_geo, 'wb')
    file.write(pickle.dumps(geo_sidecar))
    
    results = {
        'blockaverage': blockaverage,
        'epochs': epochs
        }
    
    # SAVE data a pickle for now  # !!! Change to snirf in future when its debugged
    with open(out_pkl, "wb") as f:        # if output is a single string, it wraps it in an output object and need to index in
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # # SAVE data as netcdf in addition to snirf
    # blockaverage.to_netcdf(path=out_blkavg_nc)
    # blockaverage.close()
    
    # epochs.to_netcdf(path=out_epoch_nc)
    # epochs.close()
    
    print("Block average data saved successfully")
    
    # Flatten list of bad channels and take only unique chan values
    idx_sat_flat = [x for xs in idx_sat_runs for x in xs] # flatten list of bad chans indices for all runs
    idx_amp_flat = [x for xs in idx_amp_runs for x in xs]
    bad_chans_sat_flat = [x for xs in bad_chans_sat_runs for x in xs]
    bad_chans_amp_flat = [x for xs in bad_chans_amp_runs for x in xs]
    
    idx_sat = list(set(idx_sat_flat)) # get unique channel values only # !!! FIXME: want to not mark a chan bad thats only bad in 1 run in future
    idx_amp = list(set(idx_amp_flat))
    bad_chans_sat = list(set(bad_chans_sat_flat))
    bad_chans_amp = list(set(bad_chans_amp_flat))
    
    data_quality = {       
        "idx_sat": idx_sat,
        "bad_chans_sat": bad_chans_sat,
        "idx_amp": idx_amp,
        "bad_chans_amp": bad_chans_amp
        }
    
    # SAVE data quality dict as a sidecar json file   # !!! change to just keeping in xarray as a dim?
    with open(out_json, 'w') as fp:
        json.dump(data_quality, fp)
    
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

    config = snakemake.config
    
    cfg_dataset = snakemake.params.cfg_dataset
    cfg_blockaverage = snakemake.params.cfg_blockaverage
    cfg_hrf = snakemake.params.cfg_hrf
    run_files = snakemake.input.preproc  #.preproc_runs
    data_quality_files = snakemake.input.quality
    
    out_pkl = snakemake.output.pickle
    out_json = snakemake.output.json
    out_geo = snakemake.output.geo
    #out_blkavg_nc = snakemake.output.bl_nc
    #out_epoch_nc = snakemake.output.ep_nc
    
    blockaverage_func(cfg_dataset, cfg_blockaverage, cfg_hrf, run_files, data_quality_files, out_pkl, out_json, out_geo)  #, out_blkavg_nc, out_epoch_nc)
    
   
    
if __name__ == "__main__":
    main()

