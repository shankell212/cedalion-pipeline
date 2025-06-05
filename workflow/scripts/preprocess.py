# -*- coding: utf-8 -*-
"""

This function will load all the data for the specified subject and file IDs, and preprocess the data.
This function will also create several data quality report (DQR) figures that are saved in /derivatives/plots.
The function will return the preprocessed data and a list of the filenames that were loaded, both as 
two dimensional lists [subj_idx][file_idx].
The data is returned as a recording container with the following fields:
  timeseries - the data matrices with dimensions of ('channel', 'wavelength', 'time') 
     or ('channel', 'HbO/HbR', 'time') depending on the data type. 
     The following sub-fields are included:
        'amp' - the original amplitude data slightly processed to remove negative and NaN values and to 
           apply a 3 point median filter to remove outliers.
        'amp_pruned' - the 'amp' data pruned according to the SNR, SD, and amplitude thresholds.
        'od' - the optical density data
        'od_tddr' - the optical density data after TDDR motion correction is applied
        'conc_tddr' - the concentration data obtained from 'od_tddr'
        'od_splineSG' and 'conc_splineSG' - returned if splineSG motion correction is applied (i.e. flag_do_splineSG=True)
  stim - the stimulus data with 'onset', 'duration', and 'trial_type' fields and more from the events.tsv files.
  aux_ts - the auxiliary time series data from the SNIRF files.
     In addition, the following aux sub-fields are added during pre-processing:
        'gvtd' - the global variance of the time derivative of the 'od' data.
        'gvtd_tddr' - the global variance of the time derivative of the 'od_tddr' data.
        
        
Created on Tue Jun  3 11:39:55 2025

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
import numpy as np
import pandas as pd
import pint
from cedalion.physunits import units


# import my own functions from a different directory
import sys
import module_plot_DQR as plot_dqr
import module_imu_glm_filter as imu_filt
import module_preprocess as preproc

import pdb
import yaml

#%% Load in data for current subject/task/run

# !!! change to getting paths here (?)
# change output to results in snakemake folder?

config = snakemake.config

# config_path = "C:\\Users\\shank\\Documents\\GitHub\\cedalion-pipeline\\workflow\\config\\config.yaml"

# with open(config_path, 'r') as file:
#     config = yaml.safe_load(file)

cfg_dataset = config["dataset"]
cfg_preprocess = config["preprocess"]

if not cfg_dataset['derivatives_subfolder']:
    cfg_dataset['derivatives_subfolder'] = ''


snirf_path = snakemake.input[0]
events_path = snakemake.input[1]


# snirf_path = f"{cfg_dataset['root_dir']}/sub-{cfg_dataset['subject'][0]}/nirs/sub-{cfg_dataset['subject'][0]}_task-{cfg_dataset['task'][0]}_run-{cfg_dataset['run'][0]}_nirs.snirf"
# events_path =  f"{cfg_dataset['root_dir']}/sub-{cfg_dataset['subject'][0]}/nirs/sub-{cfg_dataset['subject'][0]}_task-{cfg_dataset['task'][0]}_run-{cfg_dataset['run'][0]}_events.tsv"



# save_path = f"{cfg_dataset['root_dir']}/derivatives/{cfg_dataset['derivatives_subfolder']}/preprocessed_data/sub-{cfg_dataset['subject'][0]}/"
# save_file = f"{save_path}sub-{cfg_dataset['subject'][0]}_task-{cfg_dataset['task'][0]}_run-{cfg_dataset['run'][0]}_nirs_preprocessed.snirf"

# der_dir = os.path.join(save_path)
# if not os.path.exists(der_dir):
#     os.makedirs(der_dir)



records = cedalion.io.read_snirf( snirf_path ) 
rec = records[0]

if not os.path.exists( events_path ):  
    print( f"Error: File {events_path} does not exist" )
else:
    stim_df = pd.read_csv( events_path, sep='\t' )
    rec.stim = stim_df
    

# get filename for plots
filename = os.path.basename(snirf_path)
filnm, _ext = os.path.splitext(filename)

#%% Preprocess

for step_name, params in config["preprocess"]["steps"].items():
    print(step_name)
    print(params)
    
    
   # If this step is disabled, skip it
    if not (params.get("enable", False))  and (step_name != "prune"):
        continue
   
    if step_name == "median_filter":
        rec = preproc.median_filt( rec, params['order'] )
    
    elif step_name == "prune":
        # change from str to pint object
        cfg_preprocess["steps"]["prune"]["sd_thresh"] = [units(x) for x in cfg_preprocess["steps"]["prune"]["sd_thresh"]]
        cfg_preprocess["steps"]["prune"]["window_length"] = units(cfg_preprocess["steps"]["prune"]["window_length"])
        cfg_preprocess["steps"]["prune"]["amp_thresh"] = [float(x) if isinstance(x,str) else x for x in cfg_preprocess["steps"]["prune"]["amp_thresh"]]
        
        rec = preproc.pruneChannels( rec, params)
        chs_pruned = rec.get_mask("chs_pruned")
        pruned_chans = chs_pruned.where(chs_pruned != 0.58, drop=True).channel.values # get array of channels that were pruned


    # Calculate OD 
    # if flag pruned channels is True, then do rest of preprocessing on pruned amp, if not then do preprocessing on unpruned data
    elif step_name == "int2od":
        if cfg_preprocess['steps']['prune']['enable']:
            rec["od"] = cedalion.nirs.int2od(rec['amp_pruned'])                
        else:
            rec["od"] = cedalion.nirs.int2od(rec['amp'])
            #del rec.timeseries['amp_pruned']   # delete pruned amp from time series
        
        rec["od_corrected"] = rec["od"]
    
        
    elif step_name in ("calc_slope_b4", "calc_slope_before", "slope_b4", "slope_before"):
        # Get the slope of 'od' before motion correction and any bandpass filtering 
        slope_base = preproc.quant_slope(rec, "od", True)
        
    elif step_name in ("calc_gvtd_b4", "calc_gvtd_before", "gvtd_b4", "gvtd_before"):
        # Calculate GVTD on pruned data
        #amp_masked = preproc.prune_mask_ts(rec['amp'], pruned_chans)  # use chs_pruned to get gvtd w/out pruned data (could also zscore in gvtd func)
        rec.aux_ts["gvtd"], _ = quality.gvtd(rec['amp_pruned']) 
        rec.aux_ts["gvtd"].name = "gvtd"
    
    # Walking filter 
    elif step_name == 'imu_glm': 
        #print('Starting imu glm filtering step on walking portion of data.')
        rec["od_corrected"] = imu_filt.filterWalking(rec, "od", params, filnm, cfg_dataset) 
        
        
    #%% MOTION CORRECTION: 
    # tddr
    elif step_name in ("tddr", "motion_correct_tddr"):
        rec['od_corrected'] = motion_correct.tddr( rec['od_corrected'] )  

        slope_corrected = preproc.quant_slope(rec, "od_corrected", False)  # Get slopes after correction before bandpass filtering
        rec['od_corrected'] = rec['od_corrected'].where( ~rec['od_corrected'].isnull(), 0)  #1e-18 )  # replace any NaNs after TDDR
        
    # !!! add spline
    # if step_name in ("spline", "motion_correct_spline"):
    #     if 'od_corrected' in rec.timeseries.keys():
    #         rec['od_corrected'] = motion_correct.motion_correct_spline( rec['od_corrected'] )  
    #     else:   # do tddr on uncorrected od
    #         rec['od_corrected'] = motion_correct.motion_correct_spline( rec['od'] ) 
    
    
    # splineSG
    elif step_name in ("splineSG", "motion_correct_splineSG"):
        rec['od_corrected'] = motion_correct.motion_correct_splineSG( rec['od_corrected'], params['p'], params['frame_size'] )  
 
    
    # !!! add PCA
    # elif step_name in ("PCA", "motion_correct_PCA"):
    
    # pca recurse
    elif step_name in ("PCA_recurse", "motion_correct_PCA_recurse"):
        rec['od_corrected'] = motion_correct.motion_correct_PCA_recurse( rec['od_corrected'], params['t_motion'], 
                                                                               params['t_mask'],  params['stdev_thresh'], params['amp_thresh'],
                                                                               params['nSV'], params['maxIter'])  
    
    
    # wavelet
    elif step_name in ("wavelet", "motion_correct_wavelet"):
        rec['od_corrected'] = motion_correct.motion_correct_PCA_recurse( rec['od_corrected'], params['iqr'], 
                                                                               params['wavelet'], params['level'])
    
    
    
    # # if processing step given that does not match or exist  # !!! need to find better way to check for this 
    # else:
    #     # If you get here, that means this step is “enabled” but unrecognized.   # !!! this is only checking the names above
    #     raise ValueError(f"Unknown preprocessing step: {step_name}")
    
    

    # slope for Corrected OD before bandpass filtering  
    elif step_name in ("calc_slope_af", "calc_slope_after", "slope_after", "slope_af"):
        slope_corrected = preproc.quant_slope(rec, "od_corrected", False)  # Get slopes after correction before bandpass filtering
    
    # GVTD for Corrected OD before bandpass filtering  
    elif step_name in ("calc_gvtd_af", "calc_gvtd_after", "gvtd_after", "gvtd_af"):
        amp_corrected = rec['od_corrected'].copy()  
        amp_corrected.values = np.exp(-amp_corrected.values)
        amp_corrected_masked = preproc.prune_mask_ts(amp_corrected, pruned_chans)  # get "pruned" amp data post tddr
        rec.aux_ts['gvtd_corrected'], _ = quality.gvtd(amp_corrected_masked)  
        rec.aux_ts['gvtd_corrected'].name = 'gvtd_corrected'
    
    #%%
    
    # Bandpass filter od_tddr
    elif step_name == "freq_filter":
        # change from str to pint object
        cfg_preprocess["steps"]["freq_filter"]["fmin"] = units(cfg_preprocess["steps"]["freq_filter"]["fmin"])
        cfg_preprocess["steps"]["freq_filter"]["fmax"] = units(cfg_preprocess["steps"]["freq_filter"]["fmax"])
        
        rec['od_corrected'] = cedalion.sigproc.frequency.freq_filter(rec['od_corrected'], 
                                                                        params['fmin'], 
                                                                        params['fmax'])  
    # Convert OD to Conc
    elif step_name == "od2conc":
     
        dpf = xr.DataArray(
            [1, 1],
            dims="wavelength",
            coords={"wavelength": rec['amp'].wavelength},
        )
        rec['conc'] = cedalion.nirs.od2conc(rec['od_corrected'], rec.geo3d, dpf, spectrum="prahl")
        
    
    
    # GLM filtering step
    elif step_name == "GLM_filter":
        # change from str to pint object
        cfg_preprocess["steps"]["GLM_filter"]["distance_threshold"] = units(cfg_preprocess["steps"]["GLM_filter"]["distance_threshold"])
        cfg_preprocess["steps"]["GLM_filter"]["t_delta"] = units(cfg_preprocess["steps"]["GLM_filter"]["t_delta"])
        cfg_preprocess["steps"]["GLM_filter"]["t_std"] = units(cfg_preprocess["steps"]["GLM_filter"]["t_std"])
        config['hrf']['t_pre'] = units(config['hrf']['t_pre'])
        config['hrf']['t_post'] = units(config['hrf']['t_post'])
        
        rec = preproc.GLM(rec, 'conc', params, config['hrf'], pruned_chans) # passing in pruned channels
        
        rec['od_corrected'] = cedalion.nirs.conc2od(rec['conc'], rec.geo3d, dpf)  # Convert GLM filtered data back to OD
        rec['od_corrected'] = rec['od_corrected'].transpose('channel', 'wavelength', 'time') # need to transpose to match rec['od'] bc conc2od switches the axes
    
    
    
    # Plot DQR
    elif step_name in ("DQR_plot", "plot_DQR", "plot_dqr", "dqr_plot"):
        lambda0 = rec['amp_pruned'].wavelength[0].wavelength.values
        lambda1 = rec['amp_pruned'].wavelength[1].wavelength.values
        snr0, _ = quality.snr(rec['amp_pruned'].sel(wavelength=lambda0), cfg_preprocess['steps']["prune"]['snr_thresh'])
        snr1, _ = quality.snr(rec['amp_pruned'].sel(wavelength=lambda1), cfg_preprocess['steps']["prune"]['snr_thresh'])
    
        plot_dqr.plotDQR( rec, chs_pruned, cfg_preprocess['steps'], filnm, cfg_dataset, config['hrf'] )
        
        # if MA correction was performed, plot slope b4 and after
        #if not np.array_equal(rec["od_corrected"].data, rec["od"].data):    #rec['od_corrected'].data != rec['od'].data:  
        if not (rec["od_corrected"].data == rec["od"].data).all():
            plot_dqr.plot_slope(rec, [slope_base, slope_corrected], cfg_preprocess['steps'], filnm, cfg_dataset)    # !!! make step instead
        
        print("plotting DQR")  # !!! how to add in plot_group_DQR ?
        


# Save preprocessed data as a snirf file
cedalion.io.snirf.write_snirf(snakemake.output[0], rec)

#%%
# cedalion.io.snirf.write_snirf(save_file, rec)

