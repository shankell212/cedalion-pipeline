# -*- coding: utf-8 -*-
"""
Perform blockaverage to get HRF

Created on Thu Jun  5 09:40:42 2025

@author: shank
"""

#%% Imports

import os
import cedalion
import numpy as np
import xarray as xr
import pint
from cedalion import units
from cedalion.dataclasses.geometry import PointType
import gzip
import pickle
import json
import pandas as pd
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.join(script_dir, 'modules')
sys.path.append(modules_path)

import module_hrf_est as mhrf


#%% Block average func

def hrf_est_func(cfg_hrf, run_files, data_quality_files, out_file, event_files=None, preproc_files=None):
    """Estimate the HRF (block-average or GLM) from per-run data.

    ``run_files`` are either preprocessed snirf files (channel-space, default
    pipeline order) or parcel-space image-recon .nc files (Snakefile_reconfirst,
    where image recon runs before HRF estimation). The two are told apart by file
    extension.

    ``event_files`` (raw per-run events.tsv) is required for parcel-space input,
    since a reconstructed .nc file has no cedalion Record to carry stim info the
    way a snirf file does.

    ``preproc_files`` (the original per-run preprocessed snirf paths, channel-space)
    is required for parcel-space input only when GLM's do_short_sep is enabled --
    short-separation regression needs real channel geometry that parcel-space data
    doesn't have, so it's computed from the channel-space run and reused as a shared
    covariate across all parcels (see module_hrf_est.GLM).
    """
    print(f'run_files: {run_files}')

    # update units
    cfg_hrf['t_pre']= units(cfg_hrf['t_pre'])
    cfg_hrf['t_post']= units(cfg_hrf['t_post'])

    if cfg_hrf['GLM']['enable']:
        cfg_GLM = cfg_hrf['GLM']
        cfg_GLM["basis_func_params"]
        for param in cfg_GLM["basis_func_params"]:
            cfg_GLM["basis_func_params"][param] = units(cfg_GLM["basis_func_params"][param])
        if cfg_GLM['distance_threshold']:
            cfg_GLM['distance_threshold'] = units(cfg_GLM['distance_threshold'])

    # Loop through files
    pruned_chans_lst = []
    bad_channels_runs = []
    preproc_runs = []  # channel-space Records, only populated for parcel-space + do_short_sep GLM

    for file_idx, run in enumerate(run_files):        # loop through files and concatinate runs for GLM and epochs for blockaverage

        # if file path/ current run does not exist for this file, continue without it  (i.e. subj dropped out)
        if not os.path.isfile(run):
            continue

        is_imgspace = str(run).endswith('.nc')

        if is_imgspace:
            # Parcel-space per-run reconstruction (Snakefile_reconfirst: image recon -> HRF estimation)
            ds_run = xr.open_dataset(run)
            ts = ds_run['Xs'].pint.quantify()
            geo2d = ds_run['geo2d']
            geo3d = ds_run['geo3d']
            ds_run.close()

            geo2d = geo2d.pint.quantify().rename({'pos2d': 'pos'})
            geo2d['type'] = xr.DataArray(pd.Series(geo2d['type'].values).map(lambda s: PointType[s.split('.')[-1]]).values,
                dims=geo2d['type'].dims)
            geo3d = geo3d.pint.quantify().rename({'pos3d': 'pos'})
            geo3d['type'] = xr.DataArray(pd.Series(geo3d['type'].values).map(lambda s: PointType[s.split('.')[-1]]).values,
                dims=geo3d['type'].dims)

            if event_files is None:
                raise ValueError("event_files is required when hrf-estimating from image-recon (.nc) input.")
            stim = pd.read_csv(event_files[file_idx], sep='\t')

            # No channel-space bad/pruned-channel concept in parcel space; image recon
            # already down-weighted bad channels during reconstruction.
            pruned_channels = np.array([])
            bad_channels = np.array([])

            if cfg_hrf['GLM']['enable'] and cfg_hrf['GLM']['do_short_sep']:
                if preproc_files is None:
                    raise ValueError("preproc_files is required for do_short_sep GLM when hrf-estimating from image-recon (.nc) input.")
                preproc_records = cedalion.io.read_snirf(fname=preproc_files[file_idx], time_units='second')
                preproc_runs.append(preproc_records[0])
                ds_dq = xr.open_dataset(data_quality_files[file_idx])
                pruned_channels = ds_dq['pruned_channels'].values
                ds_dq.close()

        else:
            # Load in snirf for curr subj and run
            records = cedalion.io.read_snirf(fname = run, time_units = 'second' ) #FIXME: HARD CODED TIME UNITS
            rec = records[0]
            rec_str = cfg_hrf['rec_str']
            if rec_str not in rec.timeseries and rec_str == 'od_02' and 'od' in rec.timeseries:
                print("WARNING: hrf_estimation rec_str='od_02' is deprecated; using cleaned preprocessed 'od' instead.")
                rec_str = 'od'
            ts = rec[rec_str].copy()
            stim = rec.stim.copy() # select the stim for the given file

            # Load in data quality info for current run
            ds = xr.open_dataset(data_quality_files[file_idx])
            pruned_channels = ds['pruned_channels'].values
            bad_channels = ds['bad_channels'].values
            geo2d = ds['geo2d']
            geo3d = ds['geo3d']
            ds.close()

            geo2d = geo2d.pint.quantify().rename({'pos2d': 'pos'}) # re-cast type coord from string back to PointType enum
            geo2d['type'] = xr.DataArray(pd.Series(geo2d['type'].values).map(lambda s: PointType[s.split('.')[-1]]).values,
                dims=geo2d['type'].dims)
            geo3d = geo3d.pint.quantify().rename({'pos3d': 'pos'})
            geo3d['type'] = xr.DataArray(pd.Series(geo3d['type'].values).map(lambda s: PointType[s.split('.')[-1]]).values,
                dims=geo3d['type'].dims)

        # spatial dim is 'channel' for the default pipeline order, 'parcel' for
        # Snakefile_reconfirst (image recon already ran, ts is always molar concentration)
        spatial_dim = 'parcel' if 'parcel' in ts.dims else 'channel'
        if 'chromo' in ts.dims:
            ts = ts.transpose('chromo', spatial_dim, 'time')
        else:
            ts = ts.transpose('wavelength', spatial_dim, 'time')

        ts = ts.assign_coords(samples=('time', np.arange(len(ts.time))))
        ts['time'] = ts.time.pint.quantify(units.s) # !!! already is s? do we need this HARD CODING SECONDS. FIX IN CEDALION FUNCS

        # get the epochs
        #FIXME: ADD IF GLM OR BLOCKAVG
        epochs_tmp = ts.cd.to_epochs(
                                    stim,  # stimulus dataframe
                                    set(stim[stim.trial_type.isin(cfg_hrf['stim_lst'])].trial_type), # select events
                                    before = cfg_hrf['t_pre'],  # seconds before stimulus
                                    after = cfg_hrf['t_post'],  # seconds after stimulus
                                )
        #FIXME: IF GLM OR BLOCK
        if file_idx == 0:
            epochs_all = epochs_tmp
            all_runs = []
            all_runs.append( (ts, stim) if is_imgspace else rec )

        else:
            epochs_all = xr.concat([epochs_all, epochs_tmp], dim='epoch')  # concatenate epochs from all runs
            all_runs.append( (ts, stim) if is_imgspace else rec )


        # Concatenate all data qual stuff
        pruned_chans_lst.append(pruned_channels)
        bad_channels_runs.append(bad_channels)

        # DONE LOOP OVER FILES
    
    # Flatten list of bad channels and take only unique chan values
    bad_channels_flat = [x for xs in bad_channels_runs for x in xs]
    bad_channels_tmp = list(set(bad_channels_flat))

    # bad_chans_sat_flat = [x for xs in bad_chans_sat_runs for x in xs]
    # bad_chans_amp_flat = [x for xs in bad_chans_amp_runs for x in xs]
    # bad_chans_sat = list(set(bad_chans_sat_flat))
    # bad_chans_amp = list(set(bad_chans_amp_flat))
    

    if cfg_hrf['GLM']['enable']:
        print('Running GLM HRF estimation')
        glm_results, hrf_estimate, hrf_mse, bad_chans_mse_lst = mhrf.GLM(
            all_runs, cfg_hrf, geo3d, pruned_chans_lst,
            short_sep_runs = preproc_runs if preproc_runs else None,
        )
    else:
        print('Running Block Average HRF estimation')
        hrf_estimate, hrf_mse, bad_chans_mse_lst = mhrf.blockaverage(epochs_all, cfg_hrf)
        glm_results = None

    #weights = glm_results.sm.
    
    bad_chans_mse_flat = [x for xs in bad_chans_mse_lst for x in xs]
    bad_chans_mse = list(set(bad_chans_mse_flat))

    bad_channels_all = np.unique(np.concat([bad_channels_tmp, bad_chans_mse]))
    
    # Save results as xr dataset to netcdf file
    ds_results = xr.Dataset()
    ds_results['hrf_est'] = hrf_estimate.pint.dequantify()  # dequant to save, will re-quant in groupaverage
    ds_results['mse_t'] = hrf_mse.pint.dequantify() # dequant to save, will re-quant in groupaverage
    ds_results['bad_channels'] = xr.DataArray(bad_channels_all, dims='bad_channel')
    # geo2d/geo3d from the last processed run (same as before -- previously read via
    # rec.geo2d/rec.geo3d, which is undefined for parcel-space input since there's no
    # Record; the loop-local geo2d/geo3d are equivalent for the snirf case too, since
    # they're re-derived from the same underlying data via the same pos<->pos2d/pos3d
    # rename convention used when the dataquality sidecar was written).
    geo2d_clean = geo2d.pint.dequantify().rename({'pos': 'pos2d'}) # dequant to save, and rename pos to pos2d to avoid confusion with geo3d pos coords
    geo2d_clean['type'] = geo2d_clean['type'].astype(str) # convert type to str
    ds_results['geo2d'] = geo2d_clean
    geo3d_clean = geo3d.pint.dequantify().rename({'pos': 'pos3d'}) # dequant to save, and rename pos to pos3d to avoid confusion with geo2d pos coords
    geo3d_clean['type'] = geo3d_clean['type'].astype(str) # convert type to str
    ds_results['geo3d'] = geo3d_clean

    # SAVE AS NETCDF FILE
    ds_results.to_netcdf(out_file, mode='w')
    
    print(f"Hrf estimation data saved successfully to {out_file}!")

    


def replace_bad_vals(data_array, bad_chans_amp, bad_chans_sat, bad_chans_mse, replacement_val, trial_type):
    # Change bad values to predetermined set val

    data_array.loc[dict(trial_type=trial_type, channel=bad_chans_amp)] = replacement_val
    data_array.loc[dict(trial_type=trial_type, channel=bad_chans_sat)] = replacement_val
    data_array.loc[dict(trial_type=trial_type, channel=bad_chans_mse)] = replacement_val

    return data_array

    
#%%

def main():
    
    cfg_hrf = snakemake.params.cfg_hrf
    run_files = snakemake.input.preproc  #.preproc_runs
    data_quality_files = snakemake.input.quality
    # Only present for rules estimating HRF from image-recon output (e.g. the
    # hrf_estimation rule in Snakefile_reconfirst); absent for the default
    # Snakefile's hrf_estimation rule, which reads snirf files directly.
    event_files = getattr(snakemake.input, 'events', None)
    preproc_files = getattr(snakemake.input, 'preproc_channelspace', None)

    out_file = snakemake.output.net_hrf
    # out_json = snakemake.output.json
    # out_geo = snakemake.output.geo

    hrf_est_func(cfg_hrf, run_files, data_quality_files, out_file, event_files, preproc_files) #, out_json, out_geo)  #, out_blkavg_nc, out_epoch_nc)
    
   
    
if __name__ == "__main__":
    main()

