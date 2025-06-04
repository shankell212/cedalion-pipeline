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

import json


# import my own functions from a different directory
import sys
import module_plot_DQR as pfDAB_dqr
import module_imu_glm_filter as pfDAB_imu

import pdb


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

    # replace negative values and NaNs with a small positive value
    rec['amp'] = rec['amp'].where( rec['amp']>0, 1e-18 ) 
    rec['amp'] = rec['amp'].where( ~rec['amp'].isnull(), 1e-18 ) 

    # if first value is 1e-18 then replace with second value
    indices = np.where(rec['amp'][:,0,0] == 1e-18)
    rec['amp'][indices[0],0,0] = rec['amp'][indices[0],0,1]
    indices = np.where(rec['amp'][:,1,0] == 1e-18)
    rec['amp'][indices[0],1,0] = rec['amp'][indices[0],1,1]

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

    amp_thresh = cfg_prune['amp_thresh']
    snr_thresh = cfg_prune['snr_thresh']
    sd_thresh = cfg_prune['sd_thresh']

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

    #i initialize chs_pruned to 0.58 (good chans)
    chs_pruned[:] = 0.58      # good snr   # was 0.4

    # get indices for where snr_mask = false
    snr_mask_false = np.where(snr_mask == False)[0]
    chs_pruned[snr_mask_false] = 0.4 # poor snr channels      # was 0.19

    # get indices for where amp_mask_sat = false   
    amp_mask_false = np.where(amp_mask_sat == False)[0]
    chs_pruned[amp_mask_false] = 0.92 # saturated channels  # was 0.0

    # get indices for where amp_mask_low = false
    amp_mask_false = np.where(amp_mask_low == False)[0]
    chs_pruned[amp_mask_false] = 0.24  # low signal channels    # was 0.8

    # get indices for where sd_mask = false
    sd_mask_false = np.where(sd_mask == False)[0]
    chs_pruned[sd_mask_false] = 0.08 # SDS channels    # was 0.65


    # put all masks in a list
    masks = [snr_mask, sd_mask, amp_mask]

    # prune channels using the masks and the operator "all", which will keep only channels that pass all three metrics
    amp_pruned, drop_list = quality.prune_ch(rec['amp'], masks, "all", flag_drop=False)

    # record the pruned array in the record
    rec['amp_pruned'] = amp_pruned

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
        return rec, chs_pruned, sci, psp

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
    chs_pruned.loc[drop_list] = 0.76     # was 0.95

    rec.set_mask('chs_pruned', chs_pruned)
    
    return rec


def GLM(rec, rec_str, cfg_GLM, cfg_hrf, pruned_chans):
    
    # get pruned data for SSR
    rec_pruned = prune_mask_ts(rec[rec_str], pruned_chans) # !!! how is this affected when using pruned data
    
    rec_pruned = rec_pruned.where( ~rec_pruned.isnull(), 0)  #1e-18 )   # set nan to 0

    #### build design matrix
    ts_long, ts_short = cedalion.nirs.split_long_short_channels(
        rec[rec_str], rec.geo3d, distance_threshold= cfg_GLM['distance_threshold']  # !!! change to rec_pruned once NaN prob fixed
    )
    
    dm = (
    glm.design_matrix.hrf_regressors(
        rec[rec_str], rec.stim, glm.GaussianKernels(cfg_hrf['t_pre'], cfg_hrf['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
    )
    & glm.design_matrix.drift_regressors(rec[rec_str], drift_order = cfg_GLM['drift_order'])
    & glm.design_matrix.closest_short_channel_regressor(rec[rec_str], ts_short, rec.geo3d)
)

    #### fit the model 
    results = glm.fit(rec[rec_str], dm, noise_model= cfg_GLM['noise_model'])  #, max_jobs=1)

    betas = results.sm.params

    pred_all = glm.predict(rec[rec_str], betas, dm)  #, channel_wise_regressors)
    pred_all = pred_all.pint.quantify('micromolar')
    
    residual = rec[rec_str] - pred_all
    
    # prediction of all HRF regressors, i.e. all regressors that start with 'HRF '
    pred_hrf = glm.predict(
                            rec[rec_str],
                            betas.sel(regressor=betas.regressor.str.startswith("HRF ")),
                            dm ) #,
                            #channel_wise_regressors)
    
    pred_hrf = pred_hrf.pint.quantify('micromolar')
    
    rec[rec_str] = pred_hrf + residual 
    
    #### get average HRF prediction 
    rec[rec_str] = rec[rec_str].transpose('chromo', 'channel', 'time')
    rec[rec_str] = rec[rec_str].assign_coords(samples=("time", np.arange(len(rec[rec_str].time))))
    rec[rec_str]['time'] = rec[rec_str].time.pint.quantify(units.s) 
             
    return rec



def quant_slope(rec, timeseries, dequantify):
    if dequantify:
        foo = rec[timeseries].copy()
        foo = foo.pint.dequantify()
        slope = foo.polyfit(dim='time', deg=1).sel(degree=1)
    else:
        slope = rec[timeseries].polyfit(dim='time', deg=1).sel(degree=1)
        
    slope = slope.rename({"polyfit_coefficients": "slope"})
    slope = slope.assign_coords(channel = rec[timeseries].channel)
    slope = slope.assign_coords(wavelength = rec[timeseries].wavelength)

    return slope


