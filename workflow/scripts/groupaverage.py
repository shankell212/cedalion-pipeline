# -*- coding: utf-8 -*-
"""
Script that does weighted group averaging

Created on Mon Jun  9 11:48:10 2025

@author: shank
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

def groupaverage_func(cfg_dataset, cfg_blockaverage, cfg_hrf, cfg_groupaverage, flag_prune_channels, blockavg_files, data_quality_files, out):
    print(blockavg_files)
    
    n_subjects = len(blockavg_files)     # !!! will wan to put this in a log ?
    
    
    # Choose correct mse values based on if blockaveraging od or conc
    if 'chromo' in cfg_blockaverage['rec_str']:
        cfg_mse = cfg_groupaverage['mse_conc']
        cfg_mse["mse_val_for_bad_data"] = units(cfg_mse["mse_val_for_bad_data"])
        cfg_mse["mse_min_thresh"] = units(cfg_mse["mse_min_thresh"])
        cfg_mse["blockaverage_val"] = units(cfg_mse["blockaverage_val"])
    else:
        cfg_mse = cfg_groupaverage['mse_od']
        if isinstance(cfg_mse["mse_val_for_bad_data"], str):
            cfg_mse["mse_val_for_bad_data"] = float(cfg_mse["mse_val_for_bad_data"])
        if isinstance(cfg_mse["mse_min_thresh"], str):
            cfg_mse["mse_min_thresh"] = float(cfg_mse["mse_min_thresh"])
        if isinstance(cfg_mse["blockaverage_val"], str):
            cfg_mse["blockaverage_val"] = float(cfg_mse["blockaverage_val"])
    mse_amp_thresh = [float(x) if isinstance(x,str) else x for x in cfg_groupaverage['mse_amp_thresh']] # convert str to float if str
    cfg_mse['mse_amp_thresh'] = min(mse_amp_thresh) # get minimum amplitude threshold
                              
    
    # Loop over subjects
    blockaverage_subj = None
    for subj_idx, subj in enumerate(blockavg_files):
        # Load in json data qual
        with open(data_quality_files[subj_idx], 'r') as fp:
            data_quality_sub = json.load(fp)
        idx_amp = np.array(data_quality_sub['bad_chans_amp'])
        idx_sat = np.array(data_quality_sub['bad_chans_sat'])
        
        # Load in current sub's blockaverage file
        with open(subj, 'rb') as f:
            rec_blockavg = pickle.load(f)
        blockaverage = rec_blockavg['blockaverage']
        epochs = rec_blockavg['epochs']
  
        blockaverage_weighted = blockaverage.copy()
        
        n_epochs = len(epochs.epoch)
        n_chs = len(epochs.channel)

        mse_t_lst = []
        
        # Loop thru trial tpes
        for idxt, trial_type in enumerate(blockaverage.trial_type.values): 
            
            epochs_zeromean = epochs.where(epochs.trial_type == trial_type, drop=True) - blockaverage_weighted.sel(trial_type=trial_type) # zero mean data
    
            if 'chromo' in blockaverage.dims:
                foo_t = epochs_zeromean.stack(measurement=['channel','chromo']).sortby('chromo')
            else:
                foo_t = epochs_zeromean.stack(measurement=['channel','wavelength']).sortby('wavelength')
            #pdb.set_trace()
            foo_t = foo_t.transpose('measurement', 'reltime', 'epoch')  # !!! this does not have trial type?
            mse_t = (foo_t**2).sum('epoch') / (n_epochs - 1)**2 # this is squared to get variance of the mean, aka MSE of the mean
            # ^ this gets the variance  across epochs
            
            # pdb.set_trace()
            # # get the variance, correctig channels we don't trust (saturated, low amp, low var) 
            # mse_t = quality.measurement_variance( # this gets the measurement variance across time ... only good for timeseries
            #         foo_t,
            #         list_bad_channels = None, #idx_bad_channels,  # !!! where to get this?  -- before this was where mse_t = 0
            #         bad_rel_var = 1e6, # If bad_abs_var is none then it uses this value relative to maximum variance
            #         bad_abs_var = None, #cfg_blockavg['cfg_mse_conc']['mse_val_for_bad_data'],
            #         calc_covariance = False
            #     )
            
            # mse_t = blockaverage_weighted + cfg_mse['mse_min_thresh'] # set a small value to avoid dominance for low variance channels
            # blockaverage_weighted.loc[dict(channel=blockaverage_weighted.isel(channel=idx_amp).channel.data)] = cfg_mse['blockaverage_val']
            # blockaverage_weighted.loc[dict(channel=blockaverage_weighted.isel(channel=idx_sat).channel.data)] = cfg_mse['blockaverage_val']
    
            # !!! maybe have the above func have a helper func that is called variance_clean  
    
            # set bad values in mse_t to the bad value threshold
            idx_bad = np.where(mse_t == 0)[0]
            idx_bad1 = idx_bad[idx_bad<n_chs]
            idx_bad2 = idx_bad[idx_bad>=n_chs] - n_chs
            
            mse_t[idx_amp,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[idx_amp+n_chs,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[idx_sat,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[idx_sat+n_chs,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[idx_bad] = cfg_mse['mse_val_for_bad_data']
            
            channels = blockaverage_weighted.channel
            blockaverage_weighted.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_amp).channel.data)] = cfg_mse['blockaverage_val']
            blockaverage_weighted.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_sat).channel.data)] = cfg_mse['blockaverage_val']
            blockaverage_weighted.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_bad1).channel.data)] = cfg_mse['blockaverage_val']
            blockaverage_weighted.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_bad2).channel.data)] = cfg_mse['blockaverage_val']
            
            blockaverage.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_amp).channel.data)] = cfg_mse['blockaverage_val']
            blockaverage.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_sat).channel.data)] = cfg_mse['blockaverage_val']
            blockaverage.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_bad1).channel.data)] = cfg_mse['blockaverage_val']
            blockaverage.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_bad2).channel.data)] = cfg_mse['blockaverage_val']
    
    
            # !!! DO we still wanna do this?
            # set the minimum value of mse_t
            mse_t = xr.where(mse_t < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], mse_t)
            
            if 'chromo' in blockaverage.dims:
                mse_t = mse_t.unstack('measurement').transpose('chromo','channel','reltime')  
            else:
                if 'reltime' in mse_t.dims:
                    mse_t = mse_t.unstack('measurement').transpose('wavelength','channel','reltime')
                else:
                    mse_t = mse_t.unstack('measurement').transpose('wavelength','channel')
                
            mse_t = mse_t.expand_dims('trial_type')
            source_coord = blockaverage_weighted['source']
            mse_t = mse_t.assign_coords(source=('channel',source_coord.data))
            detector_coord = blockaverage_weighted['detector']
            mse_t = mse_t.assign_coords(detector=('channel',detector_coord.data))
            
            
            mse_t = mse_t.assign_coords(trial_type = [trial_type]) # assign coords to match curr trial type
            mse_t_lst.append(mse_t) # append mse_t for curr trial type to list

            # DONE LOOP OVER TRIAL TYPES
            
        mse_t_tmp = xr.concat(mse_t_lst, dim='trial_type') # concat the 2 trial types
        mse_t = mse_t_tmp # reassign the newly appended mse_t with both trial types to mse_t 

        
        # gather the blockaverage across subjects
        if blockaverage_subj is None: 
            blockaverage_subj = blockaverage
            
            # add a subject dimension and coordinate
            blockaverage_subj = blockaverage_subj.expand_dims('subj')
            blockaverage_subj = blockaverage_subj.assign_coords(subj=[cfg_dataset['subject'][subj_idx]]) # !!! will need to update when exluding subs

            blockaverage_mse_subj = mse_t.expand_dims('subj') # mse of blockaverage for each sub
            blockaverage_mse_subj = blockaverage_mse_subj.assign_coords(subj=[cfg_dataset['subject'][subj_idx]]) # !!! does snakemake give list of files in order of sub list?
            
            blockaverage_mean_weighted = blockaverage_weighted / mse_t

            blockaverage_mse_inv_mean_weighted = 1 / mse_t
            
        else:   
            blockaverage_subj_tmp = blockaverage_weighted
            blockaverage_subj_tmp = blockaverage_subj_tmp.expand_dims('subj')
            blockaverage_subj_tmp = blockaverage_subj_tmp.assign_coords(subj=[cfg_dataset['subject'][subj_idx]])
            blockaverage_subj = xr.concat([blockaverage_subj, blockaverage_subj_tmp], dim='subj')

            blockaverage_mse_subj_tmp = mse_t.expand_dims('subj')
            
            blockaverage_mse_subj_tmp = blockaverage_mse_subj_tmp.assign_coords(subj=[cfg_dataset['subject'][subj_idx]])
            blockaverage_mse_subj = xr.concat([blockaverage_mse_subj, blockaverage_mse_subj_tmp], dim='subj') # !!! this does not have trial types

            blockaverage_mean_weighted = blockaverage_mean_weighted +  blockaverage_weighted / mse_t
            blockaverage_mse_inv_mean_weighted = blockaverage_mse_inv_mean_weighted + 1/mse_t 
        
        # DONE LOOP OVER SUBJECTS
        
    # get the unweighted average
    blockaverage_mean = blockaverage_subj.mean('subj')
    
    # get the weighted average  (old)
    #blockaverage_mean_weighted = blockaverage_mean_weighted / blockaverage_mse_inv_mean_weighted
    
    # get the mean mse within subjects
    mse_mean_within_subject = 1 / blockaverage_mse_inv_mean_weighted

    # get the mse between subjects
    mse_weighted_between_subjects_tmp = (blockaverage_subj - blockaverage_mean)**2 / blockaverage_mse_subj_tmp   # was -blockaverage_mean_weighted
    mse_weighted_between_subjects = mse_weighted_between_subjects_tmp.mean('subj')
    mse_weighted_between_subjects = mse_weighted_between_subjects * mse_mean_within_subject
 
    # get the weighted average
    mse_btw_within_sum_subj = blockaverage_mse_subj + mse_weighted_between_subjects
    denom = (1/mse_btw_within_sum_subj).sum('subj')
    
    blockaverage_mean_weighted = (blockaverage_subj / mse_btw_within_sum_subj).sum('subj')
    blockaverage_mean_weighted = blockaverage_mean_weighted / denom
    
    mse_total = 1/denom
    
    total_stderr_blockaverage = np.sqrt( mse_total )
    total_stderr_blockaverage = total_stderr_blockaverage.assign_coords(trial_type=blockaverage_mean_weighted.trial_type)
    
    
    
    # !!! Do we want these plots still? Would need to also load in a rec ???  - or just load in saved geo2d and geo3d?
    # # Plot scalp plot of mean, tstat,rsme + Plot mse hist
    # for idxt, trial_type in enumerate(blockaverage_mean_weighted.trial_type.values):         
    #     plot_mean_stderr(rec, rec_str, trial_type, cfg_dataset, cfg_blockavg, blockaverage_mean_weighted, 
    #                      total_stderr_blockaverage, mse_mean_within_subject, mse_weighted_between_subjects)
    #     plot_mse_hist(rec, rec_str, trial_type, cfg_dataset, blockaverage_mse_subj, cfg_blockavg['mse_val_for_bad_data'], cfg_blockavg['mse_min_thresh'])  # !!! not sure if these r working correctly tbh
    
    if flag_prune_channels:
        blockaverage_save = blockaverage_mean
    else:
        blockaverage_save = blockaverage_mean_weighted 
    
    groupavg_results = {'group_blockaverage': blockaverage_save, # group_blockaverage  rename
               'total_stderr_blockaverage': total_stderr_blockaverage,
               'blockaverage_subj': blockaverage_subj,  # always unweighted   - load into img recon
               'blockaverage_mse_subj': blockaverage_mse_subj, # - load into img recon
               # 'geo2d' : rec.geo2d,
               # 'geo3d' : rec.geo3d     # !!! save 2d and 3d pts in blockaverage????
               }
    
    # Save data a pickle for now  # !!! Change to snirf in future when its debugged
    with open(out, "wb") as f:        # if output is a single string, it wraps it in an output object and need to index in
        pickle.dump(groupavg_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Group average data saved successfully")
    




#%%

def main():
    try:
        config = snakemake.config
        
        cfg_dataset = snakemake.params.cfg_dataset  # get params
        cfg_blockaverage = snakemake.params.cfg_blockaverage
        cfg_hrf = snakemake.params.cfg_hrf
        cfg_groupaverage = snakemake.params.cfg_groupaverage
        cfg_groupaverage['mse_amp_thresh'] = snakemake.params.mse_amp_thresh
        
        blockavg_files = snakemake.input.blockavg_subs  #.preproc_runs
        data_quality_files = snakemake.input.quality
        
        out = snakemake.output[0]
        
    except:
        #config_path = "/projectnb/nphfnirs/ns/Shannon/Code/cedalion-pipeline/workflow/config/config.yaml"
        config_path = "C:\\Users\\shank\\Documents\\GitHub\\cedalion-pipeline\\workflow\\config\\config.yaml"
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        cfg_dataset = config['dataset']
        cfg_blockaverage = config['blockaverage']
        cfg_hrf = config['hrf']
        cfg_groupaverage = config['groupaverage']
        flag_prune_channels = config['preprocess']['steps']['prune']['enable']
        cfg_groupaverage['mse_amp_thresh'] = config['preprocess']['steps']['prune']['amp_thresh']
        
        
        subjects = cfg_dataset['subject']
        task = cfg_dataset['task'][0]
                
        blockavg_dir = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "blockaverage")  #, f"sub-{subj}")
        blockavg_files = [os.path.join(blockavg_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_blockaverage.pkl") for subj in subjects ]
        data_quality_files = [os.path.join(blockavg_dir, f"sub-{subj}", f"sub-{subj}_task-{task}_nirs_dataquality.json") for subj in subjects ]
        
        
        save_path = os.path.join(cfg_dataset['root_dir'], "derivatives", cfg_dataset['derivatives_subfolder'], "groupaverage")
        out = os.path.join(save_path, f"task-{task}_nirs_groupaverage.pkl")
        
        der_dir = os.path.join(save_path)
        if not os.path.exists(der_dir):
            os.makedirs(der_dir)
            
    groupaverage_func(cfg_dataset, cfg_blockaverage, cfg_hrf, cfg_groupaverage, flag_prune_channels, blockavg_files, data_quality_files, out)

if __name__ == "__main__":
    main()
    