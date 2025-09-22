# module GLM
# functions to perform GLM

#%% Imports
import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.sigproc.frequency as frequency
import cedalion.sigproc.motion_correct as motion_correct
import cedalion.xrutils as xrutils
import cedalion.models.glm as glm
import cedalion.datasets as datasets
import xarray as xr
import matplotlib.pyplot as p
import cedalion.plots as plots
from cedalion import units
import numpy as np
import pandas as pd
from functools import reduce
import operator

import json

#%% Functions

def blockaverage(epochs_all, cfg_hrf_estimation):
    all_trial_blockaverage = None
    # Block Average
    baseline = epochs_all.sel(reltime=(epochs_all.reltime < 0)).mean('reltime')
    epochs = epochs_all - baseline  # baseline subtract
    blockaverage_ep = epochs.groupby('trial_type').mean('epoch') # mean across all epochs 
    
    epochs_zeromean = epochs - blockaverage_ep   # zeromean the epochs
    
    bad_chans_mse_lst = []
    # LOOP OVER TRIAL TYPES
    for idxt, trial_type in enumerate(cfg_hrf_estimation['stim_lst']): 
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
        
        # !!! from here, we shoudl call everythign HRF
        
        # retrieve channels where mse_t = 0
        bad_mask = mse_t.sel(trial_type=trial_type).data == 0
        bad_any = bad_mask.any(axis=1)
        bad_chans_mse = mse_t.channel[bad_any].values
    
        bad_chans_mse_lst.append(bad_chans_mse)
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
        
        if all_trial_blockaverage is None:
            all_trial_blockaverage = blockaverage
            all_trial_mse = mse_t
        else:
            all_trial_blockaverage = xr.concat([all_trial_blockaverage, blockaverage], dim='trial_type') 
            all_trial_mse = xr.concat([all_trial_mse, mse_t], dim='trial_type')

    # DONE LOOP OVER TRIAL_TYPES

    return all_trial_blockaverage, all_trial_mse, bad_chans_mse_lst


def GLM(runs, rec_str, cfg_hrf_estimation, geo3d, pruned_chans_list):
    cfg_GLM = cfg_hrf_estimation['GLM']
    # t_pre = units(cfg_hrf_estimation['t_pre'])
    # t_post = units(cfg_hrf_estimation['t_post'])
    t_pre = cfg_hrf_estimation['t_pre']
    t_post = cfg_hrf_estimation['t_post']

    stim = cfg_hrf_estimation['stim_lst']
    # 1. need to concatenate runs 
    Y_all, stim_df, runs_updated = concatenate_runs(runs, rec_str, stim)

    # 2. define design matrix

    dms = glm.design_matrix.hrf_regressors(
                                    Y_all,
                                    stim_df,
                                    glm.GaussianKernels(t_pre, t_post, cfg_GLM['t_delta'], cfg_GLM['t_std'])
                                )

    # Combine drift and short-separation regressors (if any)
    if cfg_GLM['do_drift']:
        drift_regressors = get_drift_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors)

    if cfg_GLM['do_drift_legendre']:
        drift_regressors = get_drift_legendre_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors)

    if cfg_GLM['do_short_sep']:
        ss_regressors = get_short_regressors(runs_updated, pruned_chans_list, geo3d, cfg_GLM)
        dms &= reduce(operator.and_, ss_regressors)

    dms.common = dms.common.fillna(0)

    # 3. get betas and covariance
    results = glm.fit(Y_all, dms, noise_model=cfg_GLM['noise_model']) 
    betas = results.sm.params
    cov_params = results.sm.cov_params()

    # 4. estimate HRF and MSE
    basis_hrf = glm.GaussianKernels(t_pre, t_post, cfg_GLM['t_delta'], cfg_GLM['t_std'])(Y_all)

    trial_type_list = stim_df['trial_type'].unique()

    hrf_mse_list = []
    hrf_estimate_list = []
    bad_chans_mse_lst = []

    for trial_type in trial_type_list:
        
        betas_hrf = betas.sel(regressor=betas.regressor.str.startswith(f"HRF {trial_type}"))
        hrf_estimate = estimate_HRF_from_beta(betas_hrf, basis_hrf)
        
        cov_hrf = cov_params.sel(regressor_r=cov_params.regressor_r.str.startswith(f"HRF {trial_type}"),
                            regressor_c=cov_params.regressor_c.str.startswith(f"HRF {trial_type}") 
                                    )
        hrf_mse = estimate_HRF_cov(cov_hrf, basis_hrf)

        # get bad mse channels 
        if 'chromo' in hrf_mse.dims:
            bad_mask = (hrf_mse == 0).any(dim=["time", "chromo"])
        else:
            bad_mask = (hrf_mse == 0).any(dim=["time", "wavelength"])
        bad_chans_mse = hrf_mse.channel[bad_mask].values

        #bad_mask = hrf_mse.data == 0
        #bad_any = bad_mask.any(axis=1)
        #bad_chans_mse = hrf_mse.channel[bad_any].values

        hrf_estimate = hrf_estimate.expand_dims({'trial_type': [ trial_type ] })
        hrf_mse = hrf_mse.expand_dims({'trial_type': [ trial_type ] })

        hrf_estimate_list.append(hrf_estimate)
        hrf_mse_list.append(hrf_mse)
        bad_chans_mse_lst.append(bad_chans_mse)

    hrf_estimate = xr.concat(hrf_estimate_list, dim='trial_type')
    hrf_estimate = hrf_estimate.pint.quantify('molar')

    hrf_mse = xr.concat(hrf_mse_list, dim='trial_type')
    hrf_mse = hrf_mse.pint.quantify('molar**2')

    # set universal time so that all hrfs have the same time base 
    fs = frequency.sampling_rate(runs[0][0][rec_str]).to('Hz')
    before_samples = int(np.ceil((t_pre * fs).magnitude))
    after_samples = int(np.ceil((t_post * fs).magnitude))

    dT = np.round(1 / fs, 3)  # millisecond precision
    n_timepoints = len(hrf_estimate.time)
    reltime = np.linspace(-before_samples * dT, after_samples * dT, n_timepoints)

    hrf_mse = hrf_mse.assign_coords({'time': reltime})
    hrf_mse.time.attrs['units'] = 'second'

    hrf_estimate = hrf_estimate.assign_coords({'time': reltime})
    hrf_estimate.time.attrs['units'] = 'second'

    return results, hrf_estimate, hrf_mse, bad_chans_mse_lst


def estimate_HRF_cov(cov, basis_hrf):

    basis_hrf = basis_hrf.rename({'component':'regressor_c'})
    basis_hrf = basis_hrf.assign_coords(regressor_c=cov.regressor_c.values)

    tmp = xr.dot(cov, basis_hrf, dims='regressor_c')

    tmp = tmp.rename({'regressor_r':'regressor'})
    basis_hrf = basis_hrf.rename({'regressor_c':'regressor'})

    mse_t = xr.dot(basis_hrf, tmp, dims='regressor')

    return mse_t

def estimate_HRF_from_beta(betas, basis_hrf):
        
    basis_hrf = basis_hrf.rename({'component':'regressor'})
    basis_hrf = basis_hrf.assign_coords(regressor=betas.regressor.values)

    hrf_estimate = xr.dot(betas, basis_hrf, dims='regressor')

    hrf_estimates_blcorr = hrf_estimate - hrf_estimate.sel(time = hrf_estimate.time[hrf_estimate.time<0]).mean('time')

    return hrf_estimates_blcorr

def get_drift_regressors(runs, cfg_GLM):
    
    drift_regressors = []
    i=0
    for i, run  in enumerate(runs):

        drift = glm.design_matrix.drift_regressors(run, cfg_GLM['drift_order'])
        drift.common = drift.common.assign_coords({'regressor': [f'Drift {x} run {i}' for x in range(cfg_GLM['drift_order']+1)]})
        drift_regressors.append(drift)
        
    return drift_regressors

def get_drift_legendre_regressors(runs, cfg_GLM):

    drift_regressors = []
    i=0
    for i, run  in enumerate(runs):

        drift = glm.design_matrix.drift_legendre_regressors(run, cfg_GLM['drift_order'])
        drift.common = drift.common.assign_coords({'regressor': [f'Drift {x} run {i}' for x in range(cfg_GLM['drift_order']+1)]})
        drift_regressors.append(drift)

    return drift_regressors

def get_short_regressors(runs, pruned_chans_list, geo3d, cfg_GLM):
    ss_regressors = []
    i=0
    for run, pruned_chans in zip(runs, pruned_chans_list):

        rec_pruned = prune_mask_ts(run, pruned_chans) # !!! how is this affected when using pruned data
        _, ts_short = cedalion.nirs.split_long_short_channels(
                                rec_pruned, geo3d, distance_threshold= cfg_GLM['distance_threshold']  # !!! change to rec_pruned once NaN prob fixed
                                )

        short = glm.design_matrix.average_short_channel_regressor(ts_short)
        short.common = short.common.reset_coords('samples', drop=True)
        short.common = short.common.assign_coords({'regressor': [f'short run {i}']})
        ss_regressors.append(short)
        i = i+1

    return ss_regressors

def concatenate_runs(runs, rec_str, stim):

    CURRENT_OFFSET = 0
    runs_updated = []
    stim_updated = []

    for r in runs:
        rec = r[0]
        ts = rec[rec_str]
        time = ts.time.values
        new_time = time + CURRENT_OFFSET

        ts_new = ts.copy(deep=True)
        ts_new = ts_new.pint.to('molar')
        ts_new = ts_new.assign_coords(time=new_time)

        stim = rec.stim
        stim_shift = stim.copy()
        stim_shift['onset'] += CURRENT_OFFSET

        stim_updated.append(stim_shift)
        runs_updated.append(ts_new)

        CURRENT_OFFSET = new_time[-1] + (time[1] - time[0])

    Y_all = xr.concat(runs_updated, dim='time')
    Y_all.time.attrs['units'] = units.s
    stim_df = pd.concat(stim_updated, ignore_index = True)

    return Y_all, stim_df, runs_updated


def prune_mask_ts(ts, pruned_chans):
    '''
    Function to mask pruned channels with NaN .. essentially repruning channels
    Parameters
    ----------
    ts : data array
        time series from rec[rec_str].
    pruned_chans : list or array
        list or array of channels that were pruned prior.

    Returns
    -------
    ts_masked : data array
        time series that has been "repruned" or masked with data for the pruned channels as NaN.

    '''
    mask = np.isin(ts.channel.values, pruned_chans)
    
    if ts.ndim == 3 and ts.shape[0] == len(ts.channel):
        mask_expanded = mask[:, None, None]  # (chan, wav, time)
    elif ts.ndim == 3 and ts.shape[1] == len(ts.channel):
        mask_expanded = mask[None, :, None]  # (chrom, chan, time)
    else:
        raise ValueError("Expected input shape to be either (chan, dim, time) or (dim, chan, time)")

    ts_masked = ts.where(~mask_expanded, np.nan)
    return ts_masked
