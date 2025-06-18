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
    print("group avreaging: \n")
    print(blockavg_files)
    
    n_subjects = len(blockavg_files)     # !!! will wan to put this in a log ?
    all_trial_blockaverage_mean = None
    
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
                            
    
    # # Loop over subjects
    # blockaverage_subj = None
    # for subj_idx, subj in enumerate(blockavg_files):
    #     # Load in json data qual
    #     with open(data_quality_files[subj_idx], 'r') as fp:
    #         data_quality_sub = json.load(fp)
    #     idx_amp = np.array(data_quality_sub['bad_chans_amp'])
    #     idx_sat = np.array(data_quality_sub['bad_chans_sat'])
        
    #     # Load in current sub's blockaverage file
    #     with open(subj, 'rb') as f:
    #         rec_blockavg = pickle.load(f)
    #     blockaverage = rec_blockavg['blockaverage']
    #     epochs_all = rec_blockavg['epochs']
        
    #     # Load in net cdf files
    #     #blockaverage2 = xr.load_dataarray(blockavg_files_nc)
    #     #epochs2 = xr.load_dataarray(epoch_files_nc)
        
    #     blockaverage_weighted = blockaverage.copy()
        
    #     n_epochs = len(epochs.epoch)
    #     n_chs = len(epochs.channel)

    #     mse_t_lst = []
        
    # Loop thru trial tpes
    for idxt, trial_type in enumerate(cfg_hrf['stim_lst']): 
        
        blockaverage_subj = None

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
            epochs_all = rec_blockavg['epochs']
            
            # Load in net cdf files
            #blockaverage2 = xr.load_dataarray(blockavg_files_nc)
            #epochs2 = xr.load_dataarray(epoch_files_nc)
            
            blockaverage_weighted = blockaverage.copy()
            
            n_epochs = len(epochs_all.epoch)
            n_chs = len(epochs_all.channel)

            mse_t_lst = []            

            
            # baseline correct and then get the block average across all epochs and runs for that subject
            baseline = epochs_all.sel(reltime=(epochs_all.reltime < 0)).mean('reltime')
            epochs = epochs_all - baseline
            blockaverage = epochs.groupby('trial_type').mean('epoch')
            
            blockaverage_weighted = blockaverage.copy()
            n_epochs = len(epochs.epoch)
            n_chs = len(epochs.channel)
            
            
            # de-mean the epochs
            epochs_zeromean = epochs - blockaverage
        
            if 'chromo' in blockaverage.dims:
                epochs_zeromean = epochs_zeromean.stack(measurement=['channel','chromo']).sortby('chromo')
                blockaverage = blockaverage.transpose('trial_type', 'channel', 'chromo', 'reltime')
                blockaverage_weighted = blockaverage_weighted.transpose('trial_type', 'channel', 'chromo', 'reltime')

            else:
                epochs_zeromean = epochs_zeromean.stack(measurement=['channel','wavelength']).sortby('wavelength')
                blockaverage = blockaverage.transpose('trial_type', 'channel', 'wavelength', 'reltime')
                blockaverage_weighted = blockaverage_weighted.transpose('trial_type', 'channel', 'wavelength', 'reltime')

            epochs_zeromean = epochs_zeromean.transpose('trial_type', 'measurement', 'reltime', 'epoch')            
            
            mse_t = (epochs_zeromean**2).sum('epoch') / (n_epochs - 1)**2 # this is squared to get variance of the mean, aka MSE of the mean

            # set bad values in mse_t to the bad value threshold
            idx_bad = np.where(mse_t == 0)[0]
            idx_bad1 = idx_bad[idx_bad<n_chs]
            idx_bad2 = idx_bad[idx_bad>=n_chs] - n_chs
            
            mse_t[:,idx_amp,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[:,idx_amp+n_chs,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[:,idx_sat,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[:,idx_sat+n_chs,:] = cfg_mse['mse_val_for_bad_data']
            mse_t[:,idx_bad,:] = cfg_mse['mse_val_for_bad_data']
            
            channels = blockaverage_weighted.channel
            blockaverage_weighted.loc[trial_type, channels.isel(channel=idx_amp),:,:] = cfg_mse['blockaverage_val']
            blockaverage_weighted.loc[trial_type, channels.isel(channel=idx_sat),:,:] = cfg_mse['blockaverage_val']
            blockaverage_weighted.loc[trial_type, channels.isel(channel=idx_bad1),:,:] = cfg_mse['blockaverage_val']
            blockaverage_weighted.loc[trial_type, channels.isel(channel=idx_bad2),:,:] = cfg_mse['blockaverage_val']
            

            blockaverage.loc[trial_type, channels.isel(channel=idx_amp),:,:] = cfg_mse['blockaverage_val']
            blockaverage.loc[trial_type, channels.isel(channel=idx_sat),:,:] = cfg_mse['blockaverage_val']
            blockaverage.loc[trial_type, channels.isel(channel=idx_bad1),:,:] = cfg_mse['blockaverage_val']
            blockaverage.loc[trial_type, channels.isel(channel=idx_bad2),:,:] = cfg_mse['blockaverage_val']
            
            # set the minimum value of mse_t
            if 'chromo' in epochs.dims:
                mse_t = mse_t.unstack('measurement').transpose('trial_type', 'chromo','channel','reltime')
            else:
                mse_t = mse_t.unstack('measurement').transpose('trial_type','channel','wavelength','reltime')

            source_coord = blockaverage_weighted['source']
            mse_t = mse_t.assign_coords(source=('channel',source_coord.data))
            detector_coord = blockaverage_weighted['detector']
            mse_t = mse_t.assign_coords(detector=('channel',detector_coord.data))
            
            # mse_t = xr.where(mse_t < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], mse_t)

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
            
        # get the unweighted average
        blockaverage_mean = blockaverage_subj.mean('subj')
    
        # get the mean mse within subjects
        mse_mean_within_subject = 1 / sum_mse_inv
        
        blockaverage_mean_weighted = blockaverage_mean_weighted / sum_mse_inv

        mse_weighted_between_subjects_tmp = (blockaverage_subj - blockaverage_mean_weighted)**2 / mse_subj
        mse_weighted_between_subjects = mse_weighted_between_subjects_tmp.mean('subj')
        mse_weighted_between_subjects = mse_weighted_between_subjects * mse_mean_within_subject # normalized by the within subject variances as weights
     
        # get the weighted average
        mse_btw_within_sum_subj = mse_subj + mse_weighted_between_subjects
        denom = (1/mse_btw_within_sum_subj).sum('subj')
        
        blockaverage_mean_weighted = (blockaverage_subj / mse_btw_within_sum_subj).sum('subj')
        blockaverage_mean_weighted = blockaverage_mean_weighted / denom
        
        mse_total = 1/denom
        
        total_stderr_blockaverage = np.sqrt( mse_total )
        total_stderr_blockaverage = total_stderr_blockaverage.assign_coords(trial_type=blockaverage_mean_weighted.trial_type)

        if all_trial_blockaverage_mean is None:
            
            all_trial_blockaverage_mean = blockaverage_mean
            all_trial_blockaverage_weighted_mean = blockaverage_mean_weighted
            all_trial_total_stderr = total_stderr_blockaverage
            
            all_trial_blockaverage_subj = blockaverage_subj
            all_trial_blockaverage_weighted_subj = blockaverage_subj_weighted
            all_trial_mse_subj = mse_subj 
            
        else:

            all_trial_blockaverage_mean = xr.concat([all_trial_blockaverage_mean, blockaverage_mean], dim='trial_type')
            all_trial_blockaverage_weighted_mean = xr.concat([all_trial_blockaverage_weighted_mean, blockaverage_mean_weighted], dim='trial_type')
            all_trial_total_stderr = xr.concat([all_trial_total_stderr, total_stderr_blockaverage], dim='trial_type')
            
            all_trial_blockaverage_subj = xr.concat([all_trial_blockaverage_subj, blockaverage_subj], dim='trial_type')
            all_trial_blockaverage_weighted_subj = xr.concat([all_trial_blockaverage_weighted_subj, blockaverage_subj_weighted], dim='trial_type')
            all_trial_mse_subj = xr.concat([all_trial_mse_subj, mse_subj], dim='trial_type')
    # DONE LOOP OVER TRIAL_TYPES
    
    
    
    # # !!! Do we want these plots still? Would need to also load in a rec ???  - or just load in saved geo2d and geo3d?
    # # # Plot scalp plot of mean, tstat,rsme + Plot mse hist
    # # for idxt, trial_type in enumerate(blockaverage_mean_weighted.trial_type.values):         
    # #     plot_mean_stderr(rec, rec_str, trial_type, cfg_dataset, cfg_blockavg, blockaverage_mean_weighted, 
    # #                      total_stderr_blockaverage, mse_mean_within_subject, mse_weighted_between_subjects)
    # #     plot_mse_hist(rec, rec_str, trial_type, cfg_dataset, blockaverage_mse_subj, cfg_blockavg['mse_val_for_bad_data'], cfg_blockavg['mse_min_thresh'])  # !!! not sure if these r working correctly tbh
    
    
    # blockaverage_mean, blockaverage_weighted_mean, total_stderr, blockaverage_subj, blockaverage_weighted_subj, mse_subj
    
    groupavg_results = {'group_blockaverage': all_trial_blockaverage_mean,              # weighted group avg 
                   'group_blockaverage_weighted': all_trial_blockaverage_weighted_mean,   # unweighted group aaverage
                   'total_stderr_blockaverage': all_trial_total_stderr,
                   'blockaverage_subj': all_trial_blockaverage_subj,  # always unweighted   - load into img recon
                   'blockaverage_mse_subj': all_trial_mse_subj, # - load into img recon
                   'blockaverage_weighted_subj': all_trial_blockaverage_weighted_subj,
                   #'geo2d' : rec[0][0].geo2d,
                   #'geo3d' : rec[0][0].geo3d  # !!! save 2d and 3d pts in blockaverage????
               }
    
    # Save data a pickle for now  # !!! Change to snirf in future when its debugged
    with open(out, "wb") as f:        # if output is a single string, it wraps it in an output object and need to index in
        pickle.dump(groupavg_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Group average data saved successfully")
    
    
    #return all_trial_blockaverage_mean, all_trial_blockaverage_weighted_mean, all_trial_total_stderr, all_trial_blockaverage_subj, all_trial_blockaverage_weighted_subj, all_trial_mse_subj
       
            
            
    #         epochs_zeromean = epochs.where(epochs.trial_type == trial_type, drop=True) - blockaverage_weighted.sel(trial_type=trial_type) # zero mean data
    
    #         if 'chromo' in blockaverage.dims:
    #             foo_t = epochs_zeromean.stack(measurement=['channel','chromo']).sortby('chromo')
    #         else:
    #             foo_t = epochs_zeromean.stack(measurement=['channel','wavelength']).sortby('wavelength')
    #         #pdb.set_trace()
    #         foo_t = foo_t.transpose('measurement', 'reltime', 'epoch')  # !!! this does not have trial type?
    #         mse_t = (foo_t**2).sum('epoch') / (n_epochs - 1)**2 # this is squared to get variance of the mean, aka MSE of the mean
            
    #         # ^ this gets the variance  across epochs
            
    #         # pdb.set_trace()
    #         # # get the variance, correctig channels we don't trust (saturated, low amp, low var) 
    #         # mse_t = quality.measurement_variance( # this gets the measurement variance across time ... only good for timeseries
    #         #         foo_t,
    #         #         list_bad_channels = None, #idx_bad_channels,  # !!! where to get this?  -- before this was where mse_t = 0
    #         #         bad_rel_var = 1e6, # If bad_abs_var is none then it uses this value relative to maximum variance
    #         #         bad_abs_var = None, #cfg_blockavg['cfg_mse_conc']['mse_val_for_bad_data'],
    #         #         calc_covariance = False
    #         #     )
            
    #         # mse_t = blockaverage_weighted + cfg_mse['mse_min_thresh'] # set a small value to avoid dominance for low variance channels
    #         # blockaverage_weighted.loc[dict(channel=blockaverage_weighted.isel(channel=idx_amp).channel.data)] = cfg_mse['blockaverage_val']
    #         # blockaverage_weighted.loc[dict(channel=blockaverage_weighted.isel(channel=idx_sat).channel.data)] = cfg_mse['blockaverage_val']
    
    #         # !!! maybe have the above func have a helper func that is called variance_clean  
    
    #         # set bad values in mse_t to the bad value threshold
    #         idx_bad = np.where(mse_t == 0)[0]
    #         idx_bad1 = idx_bad[idx_bad<n_chs]
    #         idx_bad2 = idx_bad[idx_bad>=n_chs] - n_chs
            
    #         mse_t[idx_amp,:] = cfg_mse['mse_val_for_bad_data']
    #         mse_t[idx_amp+n_chs,:] = cfg_mse['mse_val_for_bad_data']
    #         mse_t[idx_sat,:] = cfg_mse['mse_val_for_bad_data']
    #         mse_t[idx_sat+n_chs,:] = cfg_mse['mse_val_for_bad_data']
    #         mse_t[idx_bad] = cfg_mse['mse_val_for_bad_data']
            
    #         channels = blockaverage_weighted.channel
    #         blockaverage_weighted.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_amp).channel.data)] = cfg_mse['blockaverage_val']
    #         blockaverage_weighted.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_sat).channel.data)] = cfg_mse['blockaverage_val']
    #         blockaverage_weighted.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_bad1).channel.data)] = cfg_mse['blockaverage_val']
    #         blockaverage_weighted.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_bad2).channel.data)] = cfg_mse['blockaverage_val']
            
    #         blockaverage.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_amp).channel.data)] = cfg_mse['blockaverage_val']
    #         blockaverage.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_sat).channel.data)] = cfg_mse['blockaverage_val']
    #         blockaverage.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_bad1).channel.data)] = cfg_mse['blockaverage_val']
    #         blockaverage.loc[dict(trial_type=trial_type, channel=channels.isel(channel=idx_bad2).channel.data)] = cfg_mse['blockaverage_val']
    
    
    #         # !!! DO we still wanna do this?
    #         # set the minimum value of mse_t
    #         mse_t = xr.where(mse_t < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], mse_t)
            
    #         if 'chromo' in blockaverage.dims:
    #             mse_t = mse_t.unstack('measurement').transpose('chromo','channel','reltime')  
    #         else:
    #             mse_t = mse_t.unstack('measurement').transpose('wavelength','channel','reltime')  # !!! xrutils.other_dim
                
    #         mse_t = mse_t.expand_dims('trial_type')
    #         source_coord = blockaverage_weighted['source']
    #         mse_t = mse_t.assign_coords(source=('channel',source_coord.data))
    #         detector_coord = blockaverage_weighted['detector']
    #         mse_t = mse_t.assign_coords(detector=('channel',detector_coord.data))
            
            
    #         mse_t = mse_t.assign_coords(trial_type = [trial_type]) # assign coords to match curr trial type
    #         mse_t_lst.append(mse_t) # append mse_t for curr trial type to list

    #         # DONE LOOP OVER TRIAL TYPES
            
    #     mse_t_tmp = xr.concat(mse_t_lst, dim='trial_type') # concat the 2 trial types
    #     mse_t = mse_t_tmp # reassign the newly appended mse_t with both trial types to mse_t 

        
    #     # gather the blockaverage across subjects
    #     if blockaverage_subj is None: 
    #         blockaverage_subj = blockaverage
            
    #         # add a subject dimension and coordinate
    #         blockaverage_subj = blockaverage_subj.expand_dims('subj')
    #         blockaverage_subj = blockaverage_subj.assign_coords(subj=[cfg_dataset['subject'][subj_idx]]) # !!! will need to update when exluding subs

    #         blockaverage_mse_subj = mse_t.expand_dims('subj') # mse of blockaverage for each sub
    #         blockaverage_mse_subj = blockaverage_mse_subj.assign_coords(subj=[cfg_dataset['subject'][subj_idx]]) # !!! does snakemake give list of files in order of sub list?
            
    #         blockaverage_mean_weighted = blockaverage_weighted / mse_t

    #         blockaverage_mse_inv_mean_weighted = 1 / mse_t
            
    #     else:   
    #         blockaverage_subj_tmp = blockaverage_weighted
    #         blockaverage_subj_tmp = blockaverage_subj_tmp.expand_dims('subj')
    #         blockaverage_subj_tmp = blockaverage_subj_tmp.assign_coords(subj=[cfg_dataset['subject'][subj_idx]])
    #         blockaverage_subj = xr.concat([blockaverage_subj, blockaverage_subj_tmp], dim='subj')

    #         blockaverage_mse_subj_tmp = mse_t.expand_dims('subj')
            
    #         blockaverage_mse_subj_tmp = blockaverage_mse_subj_tmp.assign_coords(subj=[cfg_dataset['subject'][subj_idx]])
    #         blockaverage_mse_subj = xr.concat([blockaverage_mse_subj, blockaverage_mse_subj_tmp], dim='subj') # !!! this does not have trial types

    #         blockaverage_mean_weighted = blockaverage_mean_weighted +  blockaverage_weighted / mse_t
    #         blockaverage_mse_inv_mean_weighted = blockaverage_mse_inv_mean_weighted + 1/mse_t 
        
    #     # DONE LOOP OVER SUBJECTS
        
    # # get the unweighted average
    # blockaverage_mean = blockaverage_subj.mean('subj')
    
    # # get the weighted average  (old)
    # #blockaverage_mean_weighted = blockaverage_mean_weighted / blockaverage_mse_inv_mean_weighted
    
    # # get the mean mse within subjects
    # mse_mean_within_subject = 1 / blockaverage_mse_inv_mean_weighted

    # # get the mse between subjects
    # mse_weighted_between_subjects_tmp = (blockaverage_subj - blockaverage_mean)**2 / blockaverage_mse_subj_tmp   # was -blockaverage_mean_weighted
    # mse_weighted_between_subjects = mse_weighted_between_subjects_tmp.mean('subj')
    # mse_weighted_between_subjects = mse_weighted_between_subjects * mse_mean_within_subject
 
    # # get the weighted average
    # mse_btw_within_sum_subj = blockaverage_mse_subj + mse_weighted_between_subjects
    # denom = (1/mse_btw_within_sum_subj).sum('subj')
    
    # blockaverage_mean_weighted = (blockaverage_subj / mse_btw_within_sum_subj).sum('subj')
    # blockaverage_mean_weighted = blockaverage_mean_weighted / denom
    
    # mse_total = 1/denom
    
    # total_stderr_blockaverage = np.sqrt( mse_total )
    # total_stderr_blockaverage = total_stderr_blockaverage.assign_coords(trial_type=blockaverage_mean_weighted.trial_type)
    
    
    
    # # !!! Do we want these plots still? Would need to also load in a rec ???  - or just load in saved geo2d and geo3d?
    # # # Plot scalp plot of mean, tstat,rsme + Plot mse hist
    # # for idxt, trial_type in enumerate(blockaverage_mean_weighted.trial_type.values):         
    # #     plot_mean_stderr(rec, rec_str, trial_type, cfg_dataset, cfg_blockavg, blockaverage_mean_weighted, 
    # #                      total_stderr_blockaverage, mse_mean_within_subject, mse_weighted_between_subjects)
    # #     plot_mse_hist(rec, rec_str, trial_type, cfg_dataset, blockaverage_mse_subj, cfg_blockavg['mse_val_for_bad_data'], cfg_blockavg['mse_min_thresh'])  # !!! not sure if these r working correctly tbh
    
    
    # groupavg_results = {'group_blockaverage_weighted': blockaverage_mean_weighted, # weighted group avg   
    #             'group_blockaverage': blockaverage_mean,  # unweighted group aaverage
    #            'total_stderr_blockaverage': total_stderr_blockaverage,
    #            'blockaverage_subj': blockaverage_subj,  # always unweighted   - load into img recon
    #            'blockaverage_mse_subj': blockaverage_mse_subj, # - load into img recon
    #            # 'geo2d' : rec.geo2d,
    #            # 'geo3d' : rec.geo3d     # !!! save 2d and 3d pts in blockaverage????
    #            }
    
    # # Save data a pickle for now  # !!! Change to snirf in future when its debugged
    # with open(out, "wb") as f:        # if output is a single string, it wraps it in an output object and need to index in
    #     pickle.dump(groupavg_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # print("Group average data saved successfully")
    


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
    #blockavg_files_nc = snakemake.input.blockavg_nc
    #epoch_files_nc = snakemake.input.epochs_nc
    
    out = snakemake.output[0]

    groupaverage_func(cfg_dataset, cfg_blockaverage, cfg_hrf, cfg_groupaverage, flag_prune_channels, blockavg_files, data_quality_files, out)
    
            
if __name__ == "__main__":
    main()
