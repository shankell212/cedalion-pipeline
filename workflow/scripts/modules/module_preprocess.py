# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:39:59 2025

@author: shank
"""

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


#%%

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


def median_filt(rec, median_filt ):

    # apply a median filter to rec['amp'] along the time dimension
    # FIXME: this is to handle spikes that arise from the 1e-18 values inserted above or from other causes, 
    #        but this is an effective LPF. TDDR may handle this
    # Pad the data before applying the median filter
    pad_width = 1  # Adjust based on the kernel size
    padded_amp = rec['amp'].pad(time=(pad_width, pad_width), mode='edge')
    # Apply the median filter to the padded data
    filtered_padded_amp = padded_amp.rolling(time=median_filt, center=True).reduce(np.median)
    # Trim the padding after applying the filter
    rec['amp'] = filtered_padded_amp.isel(time=slice(pad_width, -pad_width))

    return rec


def pruneChannels( rec, cfg_prune ):
    ''' Function that prunes channels based on cfg params.
        *Pruned channels are not dropped, instead they are set to NaN 
        '''
    amp_thresh = [cfg_prune['amp_thresh_min'], cfg_prune['amp_thresh_max']]
    snr_thresh = cfg_prune['snr_thresh']
    sd_thresh = [cfg_prune['sd_thresh_min'], cfg_prune['sd_thresh_max']]

    amp_thresh_sat = [0., amp_thresh[1]]
    amp_thresh_low = [amp_thresh[0], 1]

    # then we calculate the masks for each metric: SNR, SD distance and mean amplitude
    snr, snr_mask = quality.snr(rec['amp'], snr_thresh)
    _, sd_mask = quality.sd_dist(rec['amp'], rec.geo3d, sd_thresh)
    _, amp_mask = quality.mean_amp(rec['amp'], amp_thresh)
    _, amp_mask_sat = quality.mean_amp(rec['amp'], amp_thresh_sat)
    _, amp_mask_low = quality.mean_amp(rec['amp'], amp_thresh_low)

    # create an xarray of channel labels with values indicated why pruned
    chs_pruned = xr.DataArray(np.zeros(rec['amp'].shape[0]), dims=["channel"], coords={"channel": rec['amp'].channel})

    # initialize chs_pruned to 0.58 (good chans)
    chs_pruned[:] = 0.58      # good snr   # was 0.4

    # get indices for where snr_mask = false
    snr_mask_false = np.where(snr_mask == False)[0]
    chs_pruned[snr_mask_false] = 0.4 # poor snr channels      # was 0.19

    # get indices for where amp_mask_sat = false   
    amp_mask_false = np.where(amp_mask_sat == False)[0]
    chs_pruned[amp_mask_false] = 0.92 # saturated channels  # was 0.0  # !!! create vars for these values

    # get indices for where amp_mask_low = false
    amp_mask_false = np.where(amp_mask_low == False)[0]
    chs_pruned[amp_mask_false] = 0.24  # low signal channels    # was 0.8

    # get indices for where sd_mask = false
    sd_mask_false = np.where(sd_mask == False)[0]
    chs_pruned[sd_mask_false] = 0.08 # SDS channels    # was 0.65


    # put all masks in a list
    masks = [snr_mask, sd_mask, amp_mask]

    # prune channels using the masks and the operator "all", which will keep only channels that pass all three metrics
    rec['amp_pruned'], drop_list = quality.prune_ch(rec['amp'], masks, "all", flag_drop=False)

    perc_time_clean_thresh = cfg_prune['perc_time_clean_thresh']
    sci_threshold = cfg_prune['sci_threshold']
    psp_threshold = cfg_prune['psp_threshold']
    window_length = cfg_prune['window_length']

    # Here we can assess the scalp coupling index (SCI) of the channels
    sci, sci_mask = quality.sci(rec['amp_pruned'], window_length, sci_threshold)

    # We can also look at the peak spectral power which takes the peak power of the cross-correlation signal between the cardiac band of the two wavelengths
    psp, psp_mask = quality.psp(rec['amp_pruned'], window_length, psp_threshold)

    # create a mask based on SCI or PSP or BOTH
    if cfg_prune['flag_use_sci'] and cfg_prune['flag_use_psp']:
        sci_x_psp_mask = sci_mask & psp_mask
    elif cfg_prune['flag_use_sci']:
        sci_x_psp_mask = sci_mask
    elif cfg_prune['flag_use_psp']:
        sci_x_psp_mask = psp_mask
    else:
        return rec, chs_pruned

    perc_time_clean = sci_x_psp_mask.sum(dim="time") / len(sci.time)
    perc_time_mask = xrutils.mask(perc_time_clean, True)
    perc_time_mask = perc_time_mask.where(perc_time_clean > perc_time_clean_thresh, False)

    # add the lambda dimension to the perc_time_mask with two entries, 760 and 850, and duplicate the existing column of data to it
    perc_time_mask = xr.concat([perc_time_mask, perc_time_mask], dim="lambda")

    # prune channels using the masks and the operator "all", which will keep only channels that pass all three metrics
    perc_time_pruned, drop_list = quality.prune_ch(rec['amp_pruned'], perc_time_mask, "all", flag_drop=False)

    # record the pruned array in the record
    rec['amp_pruned'] = perc_time_pruned

    # modify xarray of channel labels with value of 0.95 for channels that are pruned by SCI and/or PSP
    chs_pruned.loc[drop_list] = 0.76   #SCI and/or PSP  # was 0.95
    # rec.set_mask('chs_pruned', chs_pruned)
    
    return rec, chs_pruned


def GLM_filter(rec, rec_str, cfg_GLM, cfg_hrf, pruned_chans):
    
    # !!! change to have this do a weighted avg of all the short chans
    # get pruned data for SSR
    rec_pruned = prune_mask_ts(rec[rec_str], pruned_chans) # !!! how is this affected when using pruned data

    #rec_pruned = rec_pruned.where( ~rec_pruned.isnull(), 0)  #1e-18 )   # set nan to 0

    # separate long and short channels using pruned data. 
    ts_long, ts_short = cedalion.nirs.split_long_short_channels(
        rec_pruned, rec.geo3d, distance_threshold= cfg_GLM['distance_threshold']  # !!! change to rec_pruned once NaN prob fixed
    )
    
    #### build design matrix 
    dm = (
    glm.design_matrix.hrf_regressors(
        rec[rec_str], rec.stim, glm.GaussianKernels(cfg_hrf['t_pre'], cfg_hrf['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
    )
    & glm.design_matrix.drift_regressors(rec[rec_str], drift_order = cfg_GLM['drift_order'])
    & glm.design_matrix.average_short_channel_regressor(ts_short)
)

    #### fit the model 
    results = glm.fit(rec[rec_str], dm, noise_model= cfg_GLM['noise_model'])  #, max_jobs=1)

    betas = results.sm.params

    pred_all = glm.predict(rec[rec_str], betas, dm)  
    pred_all = pred_all.pint.quantify('micromolar')
    
    residual = rec[rec_str] - pred_all
    
    # prediction of all HRF regressors, i.e. all regressors that start with 'HRF '
    pred_hrf = glm.predict(
                            rec[rec_str],
                            betas.sel(regressor=betas.regressor.str.startswith("HRF ")),
                            dm ) 
                            
    pred_hrf = pred_hrf.pint.quantify('micromolar')
    
    rec[rec_str] = pred_hrf + residual 
    
    #### get average HRF prediction 
    rec[rec_str] = rec[rec_str].transpose('chromo', 'channel', 'time')
    rec[rec_str] = rec[rec_str].assign_coords(samples=("time", np.arange(len(rec[rec_str].time))))
    rec[rec_str]['time'] = rec[rec_str].time.pint.quantify(units.s) 
             
    return rec

def GLM(runs, cfg_GLM, geo3d, pruned_chans_list, stim_list):

    # 1. need to concatenate runs 
    if len(runs) > 1:
        Y_all, stim_df, runs_updated = concatenate_runs(runs, stim_list)
    else:
        Y_all = runs[0]
        stim_df = stim_list[0]
        runs_updated = runs
        
    run_unit = Y_all.pint.units
    # 2. define design matrix
    dms = glm.design_matrix.hrf_regressors(
                                    Y_all,
                                    stim_df,
                                    glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
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
    basis_hrf = glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(Y_all)

    trial_type_list = stim_df['trial_type'].unique()

    hrf_mse_list = []
    hrf_estimate_list = []

    for trial_type in trial_type_list:
        
        betas_hrf = betas.sel(regressor=betas.regressor.str.startswith(f"HRF {trial_type}"))
        hrf_estimate = estimate_HRF_from_beta(betas_hrf, basis_hrf)
        
        cov_hrf = cov_params.sel(regressor_r=cov_params.regressor_r.str.startswith(f"HRF {trial_type}"),
                            regressor_c=cov_params.regressor_c.str.startswith(f"HRF {trial_type}") 
                                    )
        hrf_mse = estimate_HRF_cov(cov_hrf, basis_hrf)

        hrf_estimate = hrf_estimate.expand_dims({'trial_type': [ trial_type ] })
        hrf_mse = hrf_mse.expand_dims({'trial_type': [ trial_type ] })

        hrf_estimate_list.append(hrf_estimate)
        hrf_mse_list.append(hrf_mse)

    hrf_estimate = xr.concat(hrf_estimate_list, dim='trial_type')
    hrf_estimate = hrf_estimate.pint.quantify(run_unit)

    hrf_mse = xr.concat(hrf_mse_list, dim='trial_type')
    hrf_mse = hrf_mse.pint.quantify(run_unit**2)

    # set universal time so that all hrfs have the same time base 
    fs = frequency.sampling_rate(runs[0]).to('Hz')
    before_samples = int(np.ceil((cfg_GLM['t_pre'] * fs).magnitude))
    after_samples = int(np.ceil((cfg_GLM['t_post'] * fs).magnitude))

    dT = np.round(1 / fs, 3)  # millisecond precision
    n_timepoints = len(hrf_estimate.time)
    reltime = np.linspace(-before_samples * dT, after_samples * dT, n_timepoints)

    hrf_mse = hrf_mse.assign_coords({'time': reltime})
    hrf_mse.time.attrs['units'] = 'second'

    hrf_estimate = hrf_estimate.assign_coords({'time': reltime})
    hrf_estimate.time.attrs['units'] = 'second'

    return results, hrf_estimate, hrf_mse


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

def concatenate_runs(runs, stim):

    CURRENT_OFFSET = 0
    runs_updated = []
    stim_updated = []

    for s, ts in zip(stim, runs):
        time = ts.time.values
        new_time = time + CURRENT_OFFSET

        ts_new = ts.copy(deep=True)
        ts_new = ts_new.pint.to('molar')
        ts_new = ts_new.assign_coords(time=new_time)

        stim_shift = s.copy()
        stim_shift['onset'] += CURRENT_OFFSET

        stim_updated.append(stim_shift)
        runs_updated.append(ts_new)

        CURRENT_OFFSET = new_time[-1] + (time[1] - time[0])

    Y_all = xr.concat(runs_updated, dim='time')
    Y_all.time.attrs['units'] = units.s
    stim_df = pd.concat(stim_updated, ignore_index = True)

    return Y_all, stim_df, runs_updated


def quant_slope(rec, timeseries):
    foo = rec[timeseries].copy()
    foo = foo.pint.dequantify()
    slope = foo.polyfit(dim='time', deg=1).sel(degree=1)

        
    slope = slope.rename({"polyfit_coefficients": "slope"})
    slope = slope.assign_coords(channel = rec[timeseries].channel)
    slope = slope.assign_coords(wavelength = rec[timeseries].wavelength)

    return slope



# %%
