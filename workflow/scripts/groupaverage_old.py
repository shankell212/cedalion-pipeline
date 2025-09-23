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


def groupaverage_func(cfg_dataset, cfg_blockaverage, cfg_hrf, cfg_groupaverage, flag_prune_channels, blockavg_files, data_quality_files, geo_files, out):
    print("group averaging: \n")
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
            cfg_mse["mse_min_thresh"] = float(eval(cfg_mse["mse_min_thresh"]))
        if isinstance(cfg_mse["blockaverage_val"], str):
            cfg_mse["blockaverage_val"] = float(cfg_mse["blockaverage_val"])
    mse_amp_thresh = [float(eval(x)) if isinstance(x,str) else x for x in cfg_groupaverage['mse_amp_thresh']] # convert str to float if str
    cfg_mse['mse_amp_thresh'] = min(mse_amp_thresh) # get minimum amplitude threshold
                            
    
    #%%
    # # # 
    # Loop thru trial tpes
    for idxt, trial_type in enumerate(cfg_hrf['stim_lst']): 
        
        # Loop over subjects
        blockaverage_subj = None
        for subj_idx, subj in enumerate(blockavg_files):
            
            
            # Load in json data qual
            with open(data_quality_files[subj_idx], 'r') as fp:
                data_quality_sub = json.load(fp)
            
            # with gzip.open(data_quality_files[subj_idx], 'rb') as f:
            #     data_quality_sub = pickle.load(f)
            
            idx_amp = np.array(data_quality_sub['idx_amp'])
            idx_sat = np.array(data_quality_sub['idx_sat'])
            bad_chans_sat = np.array(data_quality_sub['bad_chans_sat'])
            bad_chans_amp = np.array(data_quality_sub['bad_chans_amp'])
            
            # Load in current sub's blockaverage file
            with open(subj, 'rb') as f:
                results = pickle.load(f)
            blockaverage = results['blockaverage']
            epochs_tmp = results['epochs']
        
            epochs = epochs_tmp.where(epochs_tmp.trial_type == trial_type, drop=True)
            blockaverage1 = blockaverage.sel(trial_type=trial_type)  # select current trial type
            blockaverage1 = blockaverage1.expand_dims('trial_type')  # readd trial type coord
            blockaverage = blockaverage1.copy()
            
            # Load geometric positions and landmarks # !!! don't need to do for each subject in reality, but for snakemake yes?
            with gzip.open(geo_files[subj_idx], 'rb') as f:
                geo_pos = pickle.load(f)
            geo2d = geo_pos['geo2d']
            geo3d = geo_pos['geo3d']
          
            # Load in net cdf files
            #blockaverage2 = xr.load_dataarray(blockavg_files_nc)
            #epochs2 = xr.load_dataarray(epoch_files_nc)
                        
            blockaverage_weighted = blockaverage.copy()
            n_epochs = len(epochs.epoch)

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
            
            # Change bad values to predetermined set val
            bad_mask = mse_t.sel(trial_type=trial_type).data == 0
            bad_any = bad_mask.any(axis=1)
        
            bad_chans_mse = mse_t.channel[bad_any].values
            
            blockaverage_weighted.loc[dict(trial_type=trial_type, channel=bad_chans_sat)] = cfg_mse['blockaverage_val']
            blockaverage_weighted.loc[dict(trial_type=trial_type, channel=bad_chans_amp)] = cfg_mse['blockaverage_val']
            blockaverage_weighted.loc[dict(trial_type=trial_type, channel=bad_chans_mse)] = cfg_mse['blockaverage_val']
            
            blockaverage.loc[dict(trial_type=trial_type, channel=bad_chans_amp)] = cfg_mse['blockaverage_val']
            blockaverage.loc[dict(trial_type=trial_type, channel=bad_chans_sat)] = cfg_mse['blockaverage_val']
            blockaverage.loc[dict(trial_type=trial_type, channel=bad_chans_mse)] = cfg_mse['blockaverage_val']
    
    
            # set the minimum value of mse_t
            if 'chromo' in epochs.dims:
                mse_t = mse_t.unstack('measurement').transpose('trial_type', 'chromo','channel','reltime')
            else:
                mse_t = mse_t.unstack('measurement').transpose('trial_type','channel','wavelength','reltime')
            
            # replace mse_t values after unstacking
            mse_t.loc[dict(trial_type=trial_type, channel=bad_chans_sat)] = cfg_mse['mse_val_for_bad_data']
            mse_t.loc[dict(trial_type=trial_type, channel=bad_chans_amp)] = cfg_mse['mse_val_for_bad_data']
            mse_t.loc[dict(trial_type=trial_type, channel=bad_chans_mse)] = cfg_mse['mse_val_for_bad_data']
                        
            source_coord = blockaverage_weighted['source']
            mse_t = mse_t.assign_coords(source=('channel',source_coord.data))
            detector_coord = blockaverage_weighted['detector']
            mse_t = mse_t.assign_coords(detector=('channel',detector_coord.data))
            
            mse_t = xr.where(mse_t < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], mse_t)  # !!! maybe can be removed when we have the between subject mse

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
    
    
    # !!! need to add funcs to a module or at the end of this script lawl
    # !!! Do we want these plots still? Would need to also load in a rec ???  - or just load in saved geo2d and geo3d?
    # Plot scalp plot of mean, tstat,rsme + Plot mse hist
    #rec_test = cedalion.io.read_snirf("/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/sub-01/nirs/sub-01_task-STS_run-01_nirs.snirf")
    
    # for idxt, trial_type in enumerate(all_trial_blockaverage_weighted_mean.trial_type.values):         
    #     plot_mean_stderr(rec_test[0], 'amp', trial_type, cfg_dataset, cfg_blockaverage, blockaverage_mean_weighted, 
    #                      all_trial_total_stderr, mse_mean_within_subject, mse_weighted_between_subjects, geo3d)
        
    #     plot_mse_hist(rec_test[0], 'amp', trial_type, cfg_dataset, all_trial_mse_subj, cfg_mse['mse_val_for_bad_data'], cfg_mse['mse_min_thresh'])  # !!! not sure if these r working correctly tbh
    
        

    groupavg_results = {'group_blockaverage': all_trial_blockaverage_mean,              # weighted group avg 
                   'group_blockaverage_weighted': all_trial_blockaverage_weighted_mean,   # unweighted group aaverage
                   'total_stderr_blockaverage': all_trial_total_stderr,
                   'blockaverage_subj': all_trial_blockaverage_subj,  # always unweighted   - load into img recon
                   'blockaverage_mse_subj': all_trial_mse_subj, # - load into img recon
                   'blockaverage_weighted_subj': all_trial_blockaverage_weighted_subj,
                   'geo2d' : geo2d,
                   'geo3d' : geo3d,
               }
    
    # Save data a pickle for now  # !!! Change to snirf in future when its debugged
    with open(out, "wb") as f:        # if output is a single string, it wraps it in an output object and need to index in
        pickle.dump(groupavg_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Group average data saved successfully")
    
           
    

    



#%% Plot funcs
def plot_mean_stderr(rec, rec_str, trial_type, cfg_dataset, cfg_blockavg, blockaverage_mean_weighted, blockaverage_stderr_weighted, mse_mean_within_subject, mse_weighted_between_subjects, geo3d):
    # scalp_plot the mean, stderr and t-stat
    #######################################################
    
    
    blockaverage_mean_weighted_t = blockaverage_mean_weighted
    blockaverage_stderr_weighted_t = blockaverage_stderr_weighted
    mse_mean_within_subject_t = mse_mean_within_subject
    mse_weighted_between_subjects_t = mse_weighted_between_subjects
    
    if 'chromo' in blockaverage_mean_weighted_t.dims:
        n_wav_chromo = blockaverage_mean_weighted_t.chromo.size
        name_conc_od = 'conc'
    else:
        n_wav_chromo = blockaverage_mean_weighted_t.wavelength.size
        name_conc_od = 'od'

    for i_wav_chromo in range(n_wav_chromo):
        f,ax = p.subplots(2,2,figsize=(10,10))

        ax1 = ax[0,0]
        if 'reltime' in blockaverage_mean_weighted_t.dims:
            foo_da = blockaverage_mean_weighted_t.sel(reltime=slice(cfg_blockavg['trange_hrf_stat'][0], cfg_blockavg['trange_hrf_stat'][1])).mean('reltime')
        else:
            foo_da = blockaverage_mean_weighted_t
        #foo_da = foo_da[0,:,:]
        title_str = 'Mean_' + name_conc_od + '_' + trial_type
        if 'chromo' in foo_da.dims:
            foo_da_tmp = foo_da.isel(chromo=i_wav_chromo)
        else:
            foo_da_tmp = foo_da.isel(wavelength=i_wav_chromo)
        max_val = np.nanmax(np.abs(foo_da_tmp.values))
        plots.scalp_plot(
                rec[rec_str],
                geo3d,
                foo_da_tmp,
                ax1,
                cmap='jet',
                vmin=-max_val,
                vmax=max_val,
                optode_labels=False,
                title=title_str,
                optode_size=6
            )

        ax1 = ax[0,1]
        if 'reltime' in blockaverage_mean_weighted_t.dims:
            foo_numer = blockaverage_mean_weighted_t.sel(reltime=slice(cfg_blockavg['trange_hrf_stat'][0], cfg_blockavg['trange_hrf_stat'][1])).mean('reltime')
            foo_denom = blockaverage_stderr_weighted_t.sel(reltime=slice(cfg_blockavg['trange_hrf_stat'][0], cfg_blockavg['trange_hrf_stat'][1])).mean('reltime')
            foo_da = foo_numer / foo_denom
        else:
            foo_da = blockaverage_mean_weighted_t / blockaverage_stderr_weighted_t
        #foo_da = foo_da[0,:,:]
        title_str = 'T-Stat_'+ name_conc_od + '_' + trial_type
        if 'chromo' in foo_da.dims:
            foo_da_tmp = foo_da.isel(chromo=i_wav_chromo)
        else:
            foo_da_tmp = foo_da.isel(wavelength=i_wav_chromo)
        max_val = np.nanmax(np.abs(foo_da_tmp.values))
        plots.scalp_plot(
                rec[rec_str],
                geo3d,
                foo_da_tmp,
                ax1,
                cmap='jet',
                vmin=-max_val,
                vmax=max_val,
                optode_labels=False,
                title=title_str,
                optode_size=6
            )
        
        ax1 = ax[1,0]
        if 'reltime' in blockaverage_mean_weighted_t.dims:
            foo_da = mse_mean_within_subject_t.sel(reltime=slice(cfg_blockavg['trange_hrf_stat'][0], cfg_blockavg['trange_hrf_stat'][1])).mean('reltime')
        else:
            foo_da = mse_mean_within_subject_t
        #foo_da = foo_da[0,:,:]
        foo_da = foo_da**0.5
        title_str = 'log10(RMSE) within subjects ' + name_conc_od + ' ' + trial_type
        if 'chromo' in foo_da.dims:
            foo_da_tmp = foo_da.isel(chromo=i_wav_chromo)
            foo_da_tmp = foo_da_tmp.pint.dequantify()
        else:
            foo_da_tmp = foo_da.isel(wavelength=i_wav_chromo)
        foo_da_tmp = np.log10(foo_da_tmp)
        max_val = np.nanmax(foo_da_tmp.values)
        min_val = np.nanmin(foo_da_tmp.values)
        plots.scalp_plot(
                rec[rec_str],
                geo3d,
                foo_da_tmp,
                ax1,
                cmap='jet',
                vmin=min_val,
                vmax=max_val,
                optode_labels=False,
                title=title_str,
                optode_size=6
            )

        ax1 = ax[1,1]
        if 'reltime' in blockaverage_mean_weighted_t.dims:
            foo_da = mse_weighted_between_subjects_t.sel(reltime=slice(cfg_blockavg['trange_hrf_stat'][0], cfg_blockavg['trange_hrf_stat'][1])).mean('reltime')
        else:
            foo_da = mse_weighted_between_subjects_t
        #foo_da = foo_da[0,:,:]
        foo_da = foo_da**0.5
        title_str = 'log10(RMSE) between subjects ' + name_conc_od + ' ' + trial_type 
        if 'chromo' in foo_da.dims:
            foo_da_tmp = foo_da.isel(chromo=i_wav_chromo)
            foo_da_tmp = foo_da_tmp.pint.dequantify()
        else:
            foo_da_tmp = foo_da.isel(wavelength=i_wav_chromo)
        foo_da_tmp = np.log10(foo_da_tmp)
        max_val = np.nanmax(foo_da_tmp.values)
        min_val = np.nanmin(foo_da_tmp.values)
        plots.scalp_plot(
                rec[rec_str],
                geo3d,
                foo_da_tmp,
                ax1,
                cmap='jet',
                vmin=min_val,
                vmax=max_val,
                optode_labels=False,
                title=title_str,
                optode_size=6
            )
                
        # give a title to the figure and save it
        dirnm = os.path.basename(os.path.normpath(cfg_dataset['root_dir']))
        if 'chromo' in foo_da.dims:
            title_str = f"{dirnm} - {name_conc_od} {trial_type} {foo_da.chromo.values[i_wav_chromo]} ({cfg_blockavg['trange_hrf_stat'][0]} to {cfg_blockavg['trange_hrf_stat'][1]} s)"
        else:
            title_str = f"{dirnm} - {name_conc_od} {trial_type} {foo_da.wavelength.values[i_wav_chromo]:.0f}nm ({cfg_blockavg['trange_hrf_stat'][0]} to {cfg_blockavg['trange_hrf_stat'][1]} s)"
        p.suptitle(title_str)

        save_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR', 'group_weighted_avg')
        os.makedirs(save_dir, exist_ok=True)
        
        if 'chromo' in foo_da.dims:
            p.savefig( os.path.join(save_dir, f'DQR_group_weighted_avg_{name_conc_od}_{trial_type}_{foo_da.chromo.values[i_wav_chromo]}.png') )
        else:
            p.savefig( os.path.join(save_dir, f'DQR_group_weighted_avg_{name_conc_od}_{trial_type}_{foo_da.wavelength.values[i_wav_chromo]:.0f}nm.png') )
        p.close()


def plot_mse_hist(rec, rec_str, trial_type, cfg_dataset, blockaverage_mse_subj, mse_val_for_bad_data, mse_min_thresh):
    # plot the MSE histogram
    ########################################################

    blockaverage_mse_subj_t = blockaverage_mse_subj #.sel(trial_type = trial_type)
    
    f,ax = p.subplots(2,1,figsize=(6,10))

    # plot the diagonals for all subjects
    ax1 = ax[0]
    if 'reltime' in blockaverage_mse_subj_t.dims:
        foo = blockaverage_mse_subj_t.mean('reltime')
    else:
        foo = blockaverage_mse_subj_t
    
    if 'chromo' in blockaverage_mse_subj.dims:
        foo = foo.stack(measurement=['channel','chromo']).sortby('chromo')
        name_conc_od = 'conc'
    else:
        foo = foo.stack(measurement=['channel','wavelength']).sortby('wavelength')
        name_conc_od = 'od'

    n_subjects = foo.shape[0]  

    for i in range(n_subjects):
        ax1.semilogy(foo[i,:], linewidth=0.5,alpha=0.5)
    ax1.set_title('variance in the mean for all subjects')
    ax1.set_xlabel('channel')
    ax1.legend()

    # histogram the diagonals
    ax1 = ax[1]
    foo1 = np.concatenate([foo[i] for i in range(n_subjects)]) # FIXME: need to loop over files too   # was foo[i][0] not sure what [0] was for, maybe trial type?
    # check if mse_val_for_bad_data has units
    if 'chromo' in blockaverage_mse_subj.dims:
        foo1 = np.where(foo1 == 0, mse_val_for_bad_data.magnitude, foo1) # some bad data gets through. amp=1e-6, but it is missed by the check above. Only 2 channels in 9 subjects. Seems to be channel 271
    else:
        foo1 = np.where(foo1 == 0, mse_val_for_bad_data, foo1)
    ax1.hist(np.log10(foo1), bins=100)
    
    if 'chromo' in blockaverage_mse_subj.dims:
        ax1.axvline(np.log10(mse_min_thresh.magnitude), color='r', linestyle='--', label=f'cov_min_thresh={mse_min_thresh.magnitude:.2e}')
    else:
        ax1.axvline(np.log10(mse_min_thresh), color='r', linestyle='--', label=f'cov_min_thresh={mse_min_thresh:.2e}')
        
    ax1.legend()
    ax1.set_title(f'{name_conc_od} {trial_type} - histogram for all subjects of variance in the mean')
    ax1.set_xlabel('log10(cov_diag)')

    # give a title to the figure and save it
    dirnm = os.path.basename(os.path.normpath(cfg_dataset['root_dir']))
    p.suptitle(f'Data set - {dirnm}')

    save_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'DQR', 'group_weighted_avg')
    os.makedirs(save_dir, exist_ok=True)

    p.savefig( os.path.join(save_dir, f'DQR_group_mse_histogram_{name_conc_od}_{trial_type}.png') )
    p.close()



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
