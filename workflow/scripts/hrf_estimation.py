# -*- coding: utf-8 -*-
"""
Perform blockaverage to get HRF

Created on Thu Jun  5 09:40:42 2025

@author: shank
"""

#%% Imports

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

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.join(script_dir, 'modules')
sys.path.append(modules_path)

import module_hrf_est as mhrf



#%% Block average func

def hrf_est_func(cfg_dataset, cfg_hrf, run_files, data_quality_files, out_pkl, out_json, out_geo):  #, out_blkavg_nc, out_epoch_nc):
    print(f'run_files: {run_files}')
        
    # update units 
    cfg_hrf['t_pre']= units(cfg_hrf['t_pre'])
    cfg_hrf['t_post']= units(cfg_hrf['t_post'])

    if cfg_hrf['GLM']['enable']:
        cfg_GLM = cfg_hrf['GLM']
        cfg_GLM['t_std']= units(cfg_GLM['t_std'])
        cfg_GLM['t_delta']= units(cfg_GLM['t_delta'])
        cfg_GLM['distance_threshold']= units(cfg_GLM['distance_threshold'])


    # # Choose correct mse values based on if blockaveraging od or conc
    # if 'conc' in cfg_hrf['rec_str']:
    #     cfg_mse = cfg_hrf['mse_conc']
    #     cfg_mse["mse_val_for_bad_data"] = units(cfg_mse["mse_val_for_bad_data"])
    #     cfg_mse["mse_min_thresh"] = units(cfg_mse["mse_min_thresh"])
    #     cfg_mse["blockaverage_val"] = units(cfg_mse["blockaverage_val"])
    # else:
    #     cfg_mse = cfg_hrf['mse_od']
    #     if isinstance(cfg_mse["mse_val_for_bad_data"], str):
    #         cfg_mse["mse_val_for_bad_data"] = float(cfg_mse["mse_val_for_bad_data"])
    #     if isinstance(cfg_mse["mse_min_thresh"], str):
    #         cfg_mse["mse_min_thresh"] = float(eval(cfg_mse["mse_min_thresh"]))
    #     if isinstance(cfg_mse["blockaverage_val"], str):
    #         cfg_mse["blockaverage_val"] = float(cfg_mse["blockaverage_val"])
    # mse_amp_thresh = float(eval(cfg_hrf['mse_amp_thresh']))
    # cfg_mse['mse_amp_thresh'] = mse_amp_thresh
    # #mse_amp_thresh = [float(eval(x)) if isinstance(x,str) else x for x in cfg_blockaverage['mse_amp_thresh']] # convert str to float if str
    # #cfg_mse['mse_amp_thresh'] = min(mse_amp_thresh) # get minimum amplitude threshold
                            
    
    # Loop through files
    idx_sat_runs = []
    idx_amp_runs = []
    bad_chans_sat_runs = []
    bad_chans_amp_runs = []
    
    # loop through files and concatinate runs for GLM and epochs for blockaverage
    for file_idx, run in enumerate(run_files):     
        
        # if file path/ current run does not exist for this file, continue without it  (i.e. subj dropped out)
        if not os.path.isfile(run):  # !!! do not need tis check anymore?
            continue   
        
        # # Load in snirf for curr subj and run
        # records = cedalion.io.read_snirf( run ) 
        # rec = records[0]
        with gzip.open(run, 'rb') as f:
            record = pickle.load(f)
            rec = record[0]
        ts = rec[cfg_hrf['rec_str']].copy()
        stim = rec.stim.copy() # select the stim for the given file 
        
        # Load in json data qual   # !!! now is a pkl for now
        # with open(data_quality_files[file_idx], 'r') as fp:
        #     data_quality_run = json.load(fp)
        
        with gzip.open(data_quality_files[file_idx], 'rb') as f:
            data_quality_run = pickle.load(f)
            
        geo2d = data_quality_run['geo2d']
        geo3d = data_quality_run['geo3d']
        chs_pruned = data_quality_run['chs_pruned']
            
        # check if ts has dimension chromo
        if 'chromo' in ts.dims:
            ts = ts.transpose('chromo', 'channel', 'time')  # !!! try transpose(..., 'channel', 'time') to get rid of if statement
        else:
            ts = ts.transpose('wavelength', 'channel', 'time')
            
        ts = ts.assign_coords(samples=('time', np.arange(len(ts.time))))
        ts['time'] = ts.time.pint.quantify(units.s) # !!! already is s? do we need this
                
        # get the epochs
        epochs_tmp = ts.cd.to_epochs(
                                    stim,  # stimulus dataframe
                                    set(stim[stim.trial_type.isin(cfg_hrf['stim_lst'])].trial_type), # select events  
                                    before = cfg_hrf['t_pre'],  # seconds before stimulus
                                    after = cfg_hrf['t_post'],  # seconds after stimulus
                                )
        if file_idx == 0:
            epochs_all = epochs_tmp
            all_runs = []
            all_runs.append( [rec] )

        else:
            epochs_all = xr.concat([epochs_all, epochs_tmp], dim='epoch')  # concatenate epochs from all runs
            all_runs.append( [rec] )

        # Concatenate all data data qual stuff
        idx_sat_runs.append(data_quality_run['idx_sat'])
        bad_chans_sat_runs.append(data_quality_run['bad_chans_sat'])
        idx_amp_runs.append(data_quality_run['idx_amp'])
        bad_chans_amp_runs.append(data_quality_run['bad_chans_amp'])

        # DONE LOOP OVER FILES
    
    # Flatten list of bad channels and take only unique chan values
    idx_sat_flat = [x for xs in idx_sat_runs for x in xs] # flatten list of bad chans indices for all runs
    idx_amp_flat = [x for xs in idx_amp_runs for x in xs]
    bad_chans_sat_flat = [x for xs in bad_chans_sat_runs for x in xs]
    bad_chans_amp_flat = [x for xs in bad_chans_amp_runs for x in xs]

    idx_sat = list(set(idx_sat_flat)) # get unique channel values only # !!! FIXME: want to not mark a chan bad thats only bad in 1 run in future
    idx_amp = list(set(idx_amp_flat))
    bad_chans_sat = list(set(bad_chans_sat_flat))
    bad_chans_amp = list(set(bad_chans_amp_flat))
    
    '''
    # Block Average
    baseline = epochs_all.sel(reltime=(epochs_all.reltime < 0)).mean('reltime')
    epochs = epochs_all - baseline  # baseline subtract
    blockaverage_ep = epochs.groupby('trial_type').mean('epoch') # mean across all epochs 
    
    epochs_zeromean = epochs - blockaverage_ep   # zeromean the epochs
    
    # LOOP OVER TRIAL TYPES
    for idxt, trial_type in enumerate(cfg_hrf['stim_lst']): 
        epochs_zeromean_tmp = epochs_zeromean.copy()
        blockaverage_tmp = blockaverage_ep.copy()
        # select current trial type
        epochs_zeromean_tmp = epochs_zeromean_tmp.where(epochs_zeromean_tmp.trial_type == trial_type, drop=True)
        blockaverage1 = blockaverage_tmp.sel(trial_type=trial_type)  # select current trial type
        blockaverage1 = blockaverage1.expand_dims('trial_type')  # readd trial type coord
        blockaverage = blockaverage1.copy()
        
        if 'chromo' in blockaverage.dims:
            epochs_zeromean_tmp = epochs_zeromean_tmp.stack(measurement=['channel','chromo']).sortby('chromo')
            blockaverage = blockaverage.transpose('trial_type', 'channel', 'chromo', 'reltime')
        else:
            epochs_zeromean_tmp = epochs_zeromean_tmp.stack(measurement=['channel','wavelength']).sortby('wavelength')
            blockaverage = blockaverage.transpose('trial_type', 'channel', 'wavelength', 'reltime')
    
        n_epochs = len(epochs_zeromean_tmp.epoch)

        epochs_zeromean_tmp = epochs_zeromean_tmp.transpose('trial_type', 'measurement', 'reltime', 'epoch')  
        # calc mse
        mse_t = (epochs_zeromean_tmp**2).sum('epoch') / (n_epochs - 1)**2 # this is squared to get variance of the mean, aka MSE of the mean
        
        # !!! GLM stuff

        # !!! from here, we shoudl call everythign HRF
        
        # retrieve channels where mse_t = 0
        bad_mask = mse_t.sel(trial_type=trial_type).data == 0
        bad_any = bad_mask.any(axis=1)
        bad_chans_mse = mse_t.channel[bad_any].values
    
    
        # # Replace bad vals
        # blockaverage = replace_bad_vals(blockaverage, bad_chans_amp, bad_chans_sat, bad_chans_mse, cfg_mse['blockaverage_val'], trial_type)
        
        # if 'chromo' in epochs.dims:
        #     mse_t = mse_t.unstack('measurement').transpose('trial_type', 'chromo','channel','reltime')
        # else:
        #     mse_t = mse_t.unstack('measurement').transpose('trial_type','channel','wavelength','reltime')
        
        # # replace mse_t values after unstacking
        # mse_t = replace_bad_vals(blockaverage, bad_chans_amp, bad_chans_sat, bad_chans_mse, cfg_mse['mse_val_for_bad_data'], trial_type)
                    
        # source_coord = blockaverage['source']
        # mse_t = mse_t.assign_coords(source=('channel',source_coord.data))
        # detector_coord = blockaverage['detector']
        # mse_t = mse_t.assign_coords(detector=('channel',detector_coord.data))
        
        # # set the minimum value of mse_t
        # mse_t = xr.where(mse_t < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], mse_t)  # !!! maybe can be removed when we have the between subject mse
        
        
        # if all_trial_blockaverage is None:
        #     all_trial_blockaverage = blockaverage
        #     all_trial_mse = mse_t
        # else:
        #     all_trial_blockaverage = xr.concat([all_trial_blockaverage, blockaverage], dim='trial_type') 
        #     all_trial_mse = xr.concat([all_trial_mse, mse_t], dim='trial_type')
            
    # DONE LOOP OVER TRIAL_TYPES
    '''
    
    pruned_chans = chs_pruned.where(chs_pruned != 0.58, drop=True).channel.values # get array of channels that were pruned

    if cfg_GLM['enable']:
        print('Running GLM HRF estimation')
        glm_results, hrf_estimate, hrf_mse, bad_chans_mse_lst = mhrf.GLM(all_runs, cfg_hrf['rec_str'], cfg_hrf, geo3d, pruned_chans)
    else:
        print('Running Block Average HRF estimation')
        hrf_estimate, hrf_mse, bad_chans_mse_lst = mhrf.blockaverage(epochs_all, cfg_hrf)
        glm_results = None

    bad_chans_mse_flat = [x for xs in bad_chans_mse_lst for x in xs]
    bad_chans_mse = list(set(bad_chans_mse_flat))

    bad_indices = np.unique(np.concat([bad_chans_sat, bad_chans_amp, bad_chans_mse]))

    '''
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
    '''
    
    
    # Save geometric 2d and 3d positions to sidecar file
    geo_sidecar = {
        'geo2d': geo2d,
        'geo3d': geo3d
        }
    file = gzip.GzipFile(out_geo, 'wb')
    file.write(pickle.dumps(geo_sidecar))
    file.close()
    
    results = {
        'hrf_est': hrf_estimate,
        'mse_t': hrf_mse,
        'bad_indices': bad_indices,
        #'glm_results': glm_results # NOTE: can't save in pkl, must be netcdf??? -- error when loading in pkl

        }
    
    # SAVE data a pickle for now  # !!! Change to snirf in future when its debugged
    # with open(out_pkl, "wb") as f:        # if output is a single string, it wraps it in an output object and need to index in
    #     pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    file = gzip.GzipFile(out_pkl, 'wb')  # sav eas gzipped pickle file
    file.write(pickle.dumps(results))
    file.close()    
    
    # # SAVE data as netcdf in addition to snirf
    # blockaverage.to_netcdf(path=out_blkavg_nc)
    # blockaverage.close()
    
    # epochs.to_netcdf(path=out_epoch_nc)
    # epochs.close()
    
    print("Block average data saved successfully")

    data_quality = {       
        "idx_sat": idx_sat,
        "bad_chans_sat": bad_chans_sat,
        "idx_amp": idx_amp,
        "bad_chans_amp": bad_chans_amp,
        "bad_chans_mse": bad_chans_mse
        }
    
    # # SAVE data quality dict as a sidecar json file   # !!! change to just keeping in xarray as a dim?
    file = gzip.GzipFile(out_json, 'wb')
    file.write(pickle.dumps(data_quality))
    file.close()
    
    # file = gzip.GzipFile('out_sidecar', 'wb')  # save as sidecar instead of json
    # file.write(pickle.dumps(data_quality))
    
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
    


def replace_bad_vals(data_array, bad_chans_amp, bad_chans_sat, bad_chans_mse, replacement_val, trial_type):
    # Change bad values to predetermined set val

    data_array.loc[dict(trial_type=trial_type, channel=bad_chans_amp)] = replacement_val
    data_array.loc[dict(trial_type=trial_type, channel=bad_chans_sat)] = replacement_val
    data_array.loc[dict(trial_type=trial_type, channel=bad_chans_mse)] = replacement_val

    return data_array

    
#%%

def main():

    config = snakemake.config
    
    cfg_dataset = snakemake.params.cfg_dataset
    cfg_hrf = snakemake.params.cfg_hrf
    run_files = snakemake.input.preproc  #.preproc_runs
    data_quality_files = snakemake.input.quality
    
    out_pkl = snakemake.output.pickle
    out_json = snakemake.output.json
    out_geo = snakemake.output.geo
    #out_blkavg_nc = snakemake.output.bl_nc
    #out_epoch_nc = snakemake.output.ep_nc
    
    hrf_est_func(cfg_dataset, cfg_hrf, run_files, data_quality_files, out_pkl, out_json, out_geo)  #, out_blkavg_nc, out_epoch_nc)
    
   
    
if __name__ == "__main__":
    main()
