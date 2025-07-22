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
import xarray as xr
import numpy as np
import pandas as pd
import pint
from cedalion.physunits import units


# import my own functions from a different directory
import sys
#sys.path.append('scripts/modules/')
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.join(script_dir, 'modules')
sys.path.append(modules_path)

import module_plot_DQR as plot_dqr
import module_imu_glm_filter as imu_filt
import module_preprocess as preproc
import module_image_recon as img_recon 

import pdb
import yaml
import json
import pickle
import gzip
import matplotlib.pyplot as plt

#%% Load in data for current subject/task/run

def preprocess_func(config, snirf_path, events_path, cfg_dataset, cfg_preprocess, cfg_hrf, mse_amp_thresh, out_files):
    
    if not cfg_dataset['derivatives_subfolder']:   # !!! do we need this?
        cfg_dataset['derivatives_subfolder'] = ''
    
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
    
    # Replace negatives and nans with a small pos value
    rec['amp'] = rec['amp'].where( rec['amp']>0, 1e-18 ) 
    rec['amp'] = rec['amp'].where( ~rec['amp'].isnull(), 1e-18 ) 

    # if first value is 1e-18 then replace with second value
    indices = np.where(rec['amp'][:,0,0] == 1e-18)
    rec['amp'][indices[0],0,0] = rec['amp'][indices[0],0,1]
    indices = np.where(rec['amp'][:,1,0] == 1e-18)
    rec['amp'][indices[0],1,0] = rec['amp'][indices[0],1,1]
    
    #%% Preprocess
    
    for step_name, params in cfg_preprocess["steps"].items():
        
        # If this step is disabled, skip it
        if not (params.get("enable", False))  and (step_name != "prune"): 
            continue
       
        if step_name == "bs_preproc":   # !!! only for BS data 
            
            Adot, meas_list, geo3d, amp = img_recon.load_probe(params['probe_dir'], snirf_name=params['snirf_name_probe'])
            #rec['amp'] = rec['amp'].sel(channel=Adot.channel)
            chans = list(dict.fromkeys(meas_list.channel))  # select chans we care about from probe snirf meas list
            rec['amp'] = rec['amp'].sel(channel=chans) 
            
        elif step_name == "median_filter":
            rec = preproc.median_filt( rec, params['order'] )
        
        elif step_name == "prune":
            # change from str to pint object
            cfg_preprocess["steps"]["prune"]["sd_thresh"] = [units(x) for x in cfg_preprocess["steps"]["prune"]["sd_thresh"]]
            cfg_preprocess["steps"]["prune"]["window_length"] = units(cfg_preprocess["steps"]["prune"]["window_length"])
            cfg_preprocess["steps"]["prune"]["amp_thresh"] = [float(x) if isinstance(x,str) else x for x in cfg_preprocess["steps"]["prune"]["amp_thresh"]]
            
            rec, chs_pruned = preproc.pruneChannels( rec, params)
            # chs_pruned = rec.get_mask("chs_pruned") #
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
            units_od = rec["od"].pint.units
         
        elif step_name in ("calc_slope_b4", "calc_slope_before", "slope_b4", "slope_before"):
            slope_base = preproc.quant_slope(rec, "od") # Get the slope of 'od' before motion correction and bpf 
            
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
        # !!! each step could run on teh last added time series. 
        # tddr
        elif step_name in ("tddr", "motion_correct_tddr"):
            rec['od_corrected'] = motion_correct.tddr( rec['od_corrected'] )  
            rec['od_corrected'] = rec['od_corrected'].where( ~rec['od_corrected'].isnull(), 1e-18 ) # 0  # replace any NaNs after TDDR # !!! make a step?
        
            
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
        

        # slope for Corrected OD before bandpass filtering  
        elif step_name in ("calc_slope_af", "calc_slope_after", "slope_after", "slope_af"):
            slope_corrected = preproc.quant_slope(rec, "od_corrected")  # Get slopes after correction before bandpass filtering
            # !!! am i calculating slope per second instead of per min?
            
        # GVTD for Corrected OD before bandpass filtering  
        elif step_name in ("calc_gvtd_af", "calc_gvtd_after", "gvtd_after", "gvtd_af"):
            amp_corrected = rec['od_corrected'].copy()  
            amp_corrected.values = np.exp(-amp_corrected.values)
            amp_corrected_masked = preproc.prune_mask_ts(amp_corrected, pruned_chans)  # get "pruned" amp data post tddr
            rec.aux_ts['gvtd_corrected'], _ = quality.gvtd(amp_corrected_masked)  
            rec.aux_ts['gvtd_corrected'].name = 'gvtd_corrected'
        
        #%%
        
        # Bandpass filter od
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
        # !!! add choice of gauss basis func & SSR method
        elif step_name == "GLM_filter":
            # change from str to pint object
            cfg_preprocess["steps"]["GLM_filter"]["distance_threshold"] = units(cfg_preprocess["steps"]["GLM_filter"]["distance_threshold"])
            cfg_preprocess["steps"]["GLM_filter"]["t_delta"] = units(cfg_preprocess["steps"]["GLM_filter"]["t_delta"])
            cfg_preprocess["steps"]["GLM_filter"]["t_std"] = units(cfg_preprocess["steps"]["GLM_filter"]["t_std"])
            cfg_hrf['t_pre'] = units(cfg_hrf['t_pre'])
            cfg_hrf['t_post'] = units(cfg_hrf['t_post'])
            
            rec = preproc.GLM(rec, 'conc', params, cfg_hrf, pruned_chans) # passing in pruned channels
            
            rec['od_corrected'] = cedalion.nirs.conc2od(rec['conc'], rec.geo3d, dpf)  # Convert GLM filtered data back to OD
            rec['od_corrected'] = rec['od_corrected'].transpose('channel', 'wavelength', 'time') # need to transpose to match rec['od'] bc conc2od switches the axes
        
        
        
        # Plot DQR
        elif step_name in ("DQR_plot", "plot_DQR", "plot_dqr", "dqr_plot"):
            lambda0 = rec['amp_pruned'].wavelength[0].wavelength.values
            lambda1 = rec['amp_pruned'].wavelength[1].wavelength.values
            snr0, _ = quality.snr(rec['amp_pruned'].sel(wavelength=lambda0), cfg_preprocess['steps']["prune"]['snr_thresh'])
            snr1, _ = quality.snr(rec['amp_pruned'].sel(wavelength=lambda1), cfg_preprocess['steps']["prune"]['snr_thresh'])
            snr0 = np.nanmedian(snr0.values)
            snr1 = np.nanmedian(snr1.values)
        
            plot_dqr.plotDQR( rec, chs_pruned, cfg_preprocess['steps'], filnm, cfg_dataset, cfg_hrf) #, out_files['out_dqr'], out_files['out_gvtd'] )
            
            # if MA correction was performed, plot slope b4 and after
            if not (rec["od_corrected"].data == rec["od"].data).all():
                plot_dqr.plot_slope(rec, [slope_base, slope_corrected], cfg_preprocess['steps'], filnm, cfg_dataset) #, out_files['out_slope'])
            
            # !!! how to add in plot_group_DQR ?  - make separate rule?
    
        # If you get here, that means this step is enabled but unrecognized. 
        else:
            raise ValueError(f"Unknown preprocessing step: {step_name}")
            
    if not rec['od_corrected'].pint.units:
        rec['od_corrected'] = rec['od_corrected'].pint.quantify(units_od)  # make sure od has units 
        
    if isinstance(mse_amp_thresh,str):
        mse_amp_thresh = float(mse_amp_thresh)
    idx_sat = np.where(chs_pruned == 0.92)[0]
    sat_ch_coords = chs_pruned.channel[idx_sat].values  # get channel coords
    amp = rec['amp'].mean('time').min('wavelength') # take the minimum across wavelengths
    #idx_amp = np.where(amp < cfg_preprocess["steps"]["prune"]["amp_thresh"][0])[0]
    idx_amp = np.where(amp < mse_amp_thresh)[0]   # COMES FROM GROUP AVG CFG
    amp_ch_coords = chs_pruned.channel[idx_amp].values

    data_quality = {       
        "chs_pruned": chs_pruned,  # !!! cannot save matrix or xarray as json
        "idx_sat": idx_sat.tolist(),
        "bad_chans_sat": sat_ch_coords.tolist(),
        "bad_chans_amp": amp_ch_coords.tolist(),
        "idx_amp": idx_amp.tolist(),
        'slope_base': slope_base,   # for group DQR plot
        'slope_corrected': slope_corrected,  # for Group DQR plot
        'gvtd_base': rec.aux_ts['gvtd'],
        'gvtd_corrected': rec.aux_ts['gvtd_corrected'],
        'snr0': snr0,
        'snr1': snr1,
        'geo2d': rec.geo2d,
        'geo3d': rec.geo3d
        }
    
    # Save data quality dict as a sidecar json file
    # with open(out_files['out_json'], 'w') as fp:
    #     json.dump(data_quality, fp)
    file = gzip.GzipFile(out_files['out_sidecar'], 'wb')
    file.write(pickle.dumps(data_quality))
    
    # # Save preprocessed data as a snirf file
    # cedalion.io.snirf.write_snirf(out_files['out_snirf'], rec)
    file = gzip.GzipFile(out_files['out_snirf'], 'wb')
    file.write(pickle.dumps([rec]))
    
    print("Snirf file saved successfuly")



#%% 
def main():
    config = snakemake.config   # set variables to snakemake vars
    
    snirf_path = snakemake.input.snirf
    events_path = snakemake.input.events
    
    cfg_dataset = snakemake.params.cfg_dataset
    cfg_preprocess = snakemake.params.cfg_preprocess
    cfg_hrf = snakemake.params.cfg_hrf
    mse_amp_thresh = snakemake.params.mse_amp_thresh
    
    
    out_files = {
        "out_snirf" : snakemake.output.snirf,
        "out_sidecar": snakemake.output.sidecar,
        #"out_dqr": snakemake.output.dqr_plot,
        #"out_gvtd": snakemake.output.gvtd_plot,
        #"out_slope": snakemake.output.slope_plot
        }
    preprocess_func(config, snirf_path, events_path, cfg_dataset, cfg_preprocess, cfg_hrf, mse_amp_thresh, out_files)
 
    
if __name__ == "__main__":
    main()