# -*- coding: utf-8 -*-
"""
Script that does weighted group averaging

Created on Mon Jun  9 11:48:10 2025

@author: shank
"""
import os
import cedalion
from cedalion.vis.anatomy.scalp_plot import scalp_plot
from cedalion.dataclasses.geometry import PointType
from cedalion.physunits import units
import pint
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg') # non interactive backend for plotting
import matplotlib.pyplot as plt
import pandas as pd

#%%


def groupaverage_func(cfg_dataset, cfg_groupaverage, cfg_hrf, file_names, out):
    print("group averaging: \n")
    print(file_names)

    cfg_mse = cfg_groupaverage['mse']

    # Convert units in cfg
    cfg_hrf['t_pre']= units(cfg_hrf['t_pre'])
    cfg_hrf['t_post']= units(cfg_hrf['t_post'])

    if 'conc' in cfg_hrf['rec_str']:
        cfg_mse["mse_val_for_bad_data"] = units(cfg_mse["mse_val_for_bad_data"])
        cfg_mse["mse_min_thresh"] = units(cfg_mse["mse_min_thresh"])
        cfg_mse["hrf_val"] = units(cfg_mse["hrf_val"])
    else:
        if isinstance(cfg_mse["mse_val_for_bad_data"], str):
            cfg_mse["mse_val_for_bad_data"] = float(cfg_mse["mse_val_for_bad_data"])
        if isinstance(cfg_mse["mse_min_thresh"], str):
            cfg_mse["mse_min_thresh"] = float(eval(cfg_mse["mse_min_thresh"]))
        if isinstance(cfg_mse["hrf_val"], str):
            cfg_mse["hrf_val"] = float(cfg_mse["hrf_val"])
    
    #%%
    # # # 
    # Loop thru trial tpes
    all_trial_groupaverage = None
    for trial_type in cfg_hrf['stim_lst']: 
        
        # Loop over subjects
        # hrf_est_subj = None

        all_subj_hrf_est = []
        all_subj_mse= []
        for subj in file_names:
            # Load in hrf estimation & mse for current subject
            results = xr.open_dataset(subj) # load in data

            geo2d = results['geo2d'] # grab geometry vals
            geo3d = results['geo3d']
            geo2d = geo2d.pint.quantify().rename({'pos2d': 'pos'}) # re-cast type coord from string back to PointType enum 
            geo2d['type'] = xr.DataArray(pd.Series(geo2d['type'].values).map(lambda s: PointType[s.split('.')[-1]]).values,
                dims=geo2d['type'].dims)
            geo3d = geo3d.pint.quantify().rename({'pos3d': 'pos'})
            geo3d['type'] = xr.DataArray(pd.Series(geo3d['type'].values).map(lambda s: PointType[s.split('.')[-1]]).values,
                dims=geo3d['type'].dims)

            if 'hrf' in file_names[0]:  # if hrf variable names are this
                # "corrected" selects mse_t_corrected (reweighted by image-recon
                # posterior variance, see module_hrf_est.GLM), only present when
                # hrf_estimation ran the GLM on parcel-space (reconfirst) input.
                mse_var = 'mse_t_corrected' if cfg_groupaverage.get('mse_source') == 'corrected' else 'mse_t'
                hrf_est_tmp = results['hrf_est'].pint.quantify()
                mse_t_tmp = results[mse_var].pint.quantify()
            else:
                hrf_est_tmp = results['Xs'].pint.quantify()
                mse_t_tmp = results['X_mse'].pint.quantify()
            if 'bad_channels' in results.keys(): # this is only results for hrf_est not img recon?
                bad_channels = results['bad_channels']
               

            # select current trial type
            hrf_est = hrf_est_tmp.sel(trial_type=trial_type).expand_dims('trial_type')  # select current trial type and ad back trial type as dim
            mse_t = mse_t_tmp.sel(trial_type=trial_type).expand_dims('trial_type')  
            
            if 'vertex' not in hrf_est.dims and 'parcel' not in hrf_est.dims:  # if group averaging chan space data, change values of bad chans (img recon does this)
                hrf_est.loc[dict(channel=bad_channels)] = cfg_mse['hrf_val']
                mse_t.loc[dict(channel=bad_channels)] = cfg_mse['mse_val_for_bad_data']  
                mse_t = xr.where(mse_t < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], mse_t)  # !!! maybe can be removed when we have the between subject mse
                
            # make units the same
            target_units = hrf_est.pint.units
            mse_t = mse_t.pint.to(target_units**2)
    
            all_subj_hrf_est.append(hrf_est)
            all_subj_mse.append(mse_t)

        # DONE LOOP OVER SUBJECTS
        all_subj_hrf_est_xr = xr.concat(all_subj_hrf_est, dim='subj')
        all_subj_mse_xr = xr.concat(all_subj_mse, dim='subj')

        all_subj_hrf_est_tmp = all_subj_hrf_est_xr.where(~np.isnan(all_subj_hrf_est_xr), drop=True) # drop any dim that is all NaN
        all_subj_mse_tmp = all_subj_mse_xr.where(~np.isnan(all_subj_mse_xr), drop=True)# drop any dim that is all NaN (i.e. for pruned channels)

        mse_mean_within_subj =  1 / (1 / all_subj_mse_tmp).sum('subj') # mean within subject variance

        groupaverage_unweighted = all_subj_hrf_est_tmp.mean('subj', skipna=True) # unweighted group average
        w = 1/all_subj_mse_tmp  # fixed effect weights 
        x = all_subj_hrf_est_tmp.copy() 

        # first round wted average (to calc between subj mse)
        mu_1 = (x * w).sum('subj') / w.sum('subj') # fixed effect mean

        Q = ( w * (x - mu_1)**2 ).sum('subj')  # heterogenuity statistic Q (for DL estimation of between subject variance)
        C = w.sum('subj') - ( (w**2).sum('subj') / w.sum('subj') ) # scaling factor; corrects for unequal weights.
        N = w.sizes['subj']

        # DerSimonian-Laird estimator for tau^2 (btwn subject variance)
        tau = ((Q - (N - 1)) / C).clip(min=0)
        w_2 = 1 / (tau + all_subj_mse_tmp) # random effects weights (using both within and between subject variance)

        mu_2 = (w_2 * x).sum('subj') / w_2.sum('subj') # updated weighted group average with new weights
        mse_total = 1 / w_2.sum('subj') # overall variance of the group mean 

        std_err = np.sqrt( mse_total ) 
        tstat = mu_2 / std_err


        if all_trial_groupaverage is None:
            all_trial_groupaverage = groupaverage_unweighted
            all_trial_groupaverage_weighted = mu_2
            all_trial_total_stderr = std_err
            all_trial_tstat = tstat
            all_trial_mse_total = mse_total
            all_trial_mse_weighted_between_subj = tau
            all_trial_mse_mean_within_subj = mse_mean_within_subj
            all_trial_all_subj_mse_within = all_subj_mse_tmp
        else:
            all_trial_groupaverage = xr.concat([all_trial_groupaverage, groupaverage_unweighted], dim="trial_type")
            all_trial_groupaverage_weighted = xr.concat([all_trial_groupaverage_weighted, mu_2], dim="trial_type")
            all_trial_total_stderr = xr.concat([all_trial_total_stderr, std_err], dim="trial_type")
            all_trial_tstat = xr.concat([all_trial_tstat, tstat], dim="trial_type")
            all_trial_mse_total = xr.concat([all_trial_mse_total, mse_total], dim="trial_type")
            all_trial_mse_weighted_between_subj = xr.concat([all_trial_mse_weighted_between_subj, tau], dim="trial_type")
            all_trial_mse_mean_within_subj = xr.concat([all_trial_mse_mean_within_subj, mse_mean_within_subj], dim="trial_type")
            all_trial_all_subj_mse_within = xr.concat([all_trial_all_subj_mse_within, all_subj_mse_tmp], dim="trial_type")
    # DONE LOOP OVER TRIAL_TYPES
    
    geo2d_clean = geo2d.pint.dequantify().rename({'pos': 'pos2d'}) # dequant to save, and rename pos to pos2d to avoid confusion with geo3d pos coords
    geo2d_clean['type'] = geo2d_clean['type'].astype(str) # convert type to str
    geo3d_clean = geo3d.pint.dequantify().rename({'pos': 'pos3d'}) # dequant to save, and rename pos to pos3d to avoid confusion with geo2d pos coords
    geo3d_clean['type'] = geo3d_clean['type'].astype(str) # convert type to str

    groupavg_results = {'group_average': all_trial_groupaverage.pint.dequantify(),              # unweighted group avg 
                    'group_average_weighted': all_trial_groupaverage_weighted.pint.dequantify(),   # weighted group aaverage
                    'total_stderr': all_trial_total_stderr.pint.dequantify(),  # noise
                    'tstat' : all_trial_tstat.pint.dequantify(),  # tstat of group average
                    'mse_total_group': all_trial_mse_total.pint.dequantify(),  # total variance of group average
                    'mse_weighted_btwn_subjs': all_trial_mse_weighted_between_subj.pint.dequantify(), # between subject variance,
                    'mse_mean_within_subj': all_trial_mse_mean_within_subj.pint.dequantify(),  # mean within subject variance  1/sum_mse_inv
                    'mse_all_subj_within': all_trial_all_subj_mse_within.pint.dequantify(),  # all within subject variances for all subjects and trial types (for histogram)
                    'geo2d' : geo2d_clean,
                    'geo3d' : geo3d_clean,    
               }
    
    # save result as xr dataset to netcdf file
    groupavg_dataset = xr.Dataset(groupavg_results)
    
    groupavg_dataset.to_netcdf(out, mode='w')  # write mode to overwrite if file already exists
    
    print(f"Group average data saved successfully to {out}!")
   
    # Plot scalp plot of mean, tstat, rsme + Plot mse hist  
    if 'hrf' in file_names[0]:  # if hrf variable names are this
        for trial_type in all_trial_groupaverage_weighted.trial_type.values:    
            plot_mean_stderr(hrf_est_tmp, trial_type, cfg_dataset, cfg_hrf, all_trial_groupaverage_weighted, 
                            all_trial_total_stderr, all_trial_mse_mean_within_subj, all_trial_mse_weighted_between_subj, cfg_mse['mse_val_for_bad_data'], geo3d)
            plot_mse_hist(trial_type, cfg_dataset, all_trial_all_subj_mse_within, cfg_mse['mse_val_for_bad_data'], cfg_mse['mse_min_thresh'])  
            

#%% Plot funcs
def plot_mean_stderr(ts, trial_type, cfg_dataset, cfg_blockavg, groupaverage_weighted, hrf_est_stderr_weighted, mse_mean_within_subject, mse_weighted_between_subjects, mse_val_for_bad_data,geo3d):
    # scalp_plot the mean, stderr and t-stat
    #######################################################
    middle = (cfg_blockavg['t_pre'] + cfg_blockavg['t_post']) / 2
    middle = int(middle.magnitude)
    half_width = (cfg_blockavg['t_post'] - cfg_blockavg['t_pre']) / 2
    half_width = int(half_width.magnitude)
    lower = middle - half_width
    upper = middle + half_width

    groupaverage_weighted_t = groupaverage_weighted
    hrf_est_stderr_weighted_t = hrf_est_stderr_weighted
    mse_mean_within_subject_t = mse_mean_within_subject
    mse_weighted_between_subjects_t = mse_weighted_between_subjects
    
    if 'chromo' in groupaverage_weighted_t.dims:
        n_wav_chromo = groupaverage_weighted_t.chromo.size
        name_conc_od = 'conc'
    else:
        n_wav_chromo = groupaverage_weighted_t.wavelength.size
        name_conc_od = 'od'

    for i_wav_chromo in range(n_wav_chromo):
        f,ax = plt.subplots(2,2,figsize=(10,10))

        # WEIGHTED GROUP AVG
        ax1 = ax[0,0]
        if 'time' in groupaverage_weighted_t.dims:
            foo_da = groupaverage_weighted_t.sel(time=slice(lower, upper)).mean('time')
        else:
            foo_da = groupaverage_weighted_t
        title_str = 'Mean_' + name_conc_od + '_' + trial_type
        if 'chromo' in foo_da.dims:
            foo_da_tmp = foo_da.isel(chromo=i_wav_chromo)
        else:
            foo_da_tmp = foo_da.isel(wavelength=i_wav_chromo)
        max_val = np.nanmax(np.abs(foo_da_tmp.values))
        foo_da_tmp = foo_da_tmp.sel(trial_type=trial_type)
        scalp_plot(
                ts,
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

        # T-STAT
        ax1 = ax[0,1]
        if 'time' in groupaverage_weighted_t.dims:
            foo_numer = groupaverage_weighted_t.sel(time=slice(lower, upper)).mean('time')
            foo_denom = hrf_est_stderr_weighted_t.sel(time=slice(lower, upper)).mean('time')
            foo_da = foo_numer / foo_denom
        else:
            foo_da = groupaverage_weighted_t / hrf_est_stderr_weighted_t
        title_str = 'T-Stat_'+ name_conc_od + '_' + trial_type
        if 'chromo' in foo_da.dims:
            foo_da_tmp = foo_da.isel(chromo=i_wav_chromo)
        else:
            foo_da_tmp = foo_da.isel(wavelength=i_wav_chromo)
        max_val = np.nanmax(np.abs(foo_da_tmp.values))
        foo_da_tmp = foo_da_tmp.sel(trial_type=trial_type)
        scalp_plot(
                ts,
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
        
        # RMSE WITHIN SUBJECTS
        ax1 = ax[1,0]
        if 'time' in groupaverage_weighted_t.dims:
            foo_da = mse_mean_within_subject_t.sel(time=slice(lower, upper)).mean('time')
        else:
            foo_da = mse_mean_within_subject_t
        foo_da = foo_da**0.5
        title_str = 'log10(RMSE) within subjects ' + name_conc_od + ' ' + trial_type
        if 'chromo' in foo_da.dims:
            foo_da_tmp = foo_da.isel(chromo=i_wav_chromo)
            foo_da_tmp = foo_da_tmp.pint.dequantify()
            foo_da_tmp = foo_da_tmp.where(foo_da_tmp != 0, mse_val_for_bad_data.magnitude) # remove any 0s that get through
        else:
            foo_da_tmp = foo_da.isel(wavelength=i_wav_chromo)
            foo_da_tmp = foo_da_tmp.where(foo_da_tmp != 0, mse_val_for_bad_data)
        foo_da_tmp = np.log10(foo_da_tmp)
        foo_da_tmp = foo_da_tmp.sel(trial_type=trial_type)
        max_val = np.nanmax(foo_da_tmp.values)
        min_val = np.nanmin(foo_da_tmp.values)
        scalp_plot(
                ts,
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
        
        # RMSE BETWEEN SUBJECTS
        ax1 = ax[1,1]
        if 'time' in groupaverage_weighted_t.dims:
            foo_da = mse_weighted_between_subjects_t.sel(time=slice(lower, upper)).mean('time')
        else:
            foo_da = mse_weighted_between_subjects_t
        foo_da = foo_da**0.5
        title_str = 'log10(RMSE) between subjects ' + name_conc_od + ' ' + trial_type 
        if 'chromo' in foo_da.dims:
            foo_da_tmp = foo_da.isel(chromo=i_wav_chromo)
            foo_da_tmp = foo_da_tmp.pint.dequantify()
            foo_da_tmp = foo_da_tmp.where(foo_da_tmp != 0, mse_val_for_bad_data.magnitude) # remove any 0s that get through
        else:
            foo_da_tmp = foo_da.isel(wavelength=i_wav_chromo)
            foo_da_tmp = foo_da_tmp.where(foo_da_tmp != 0, mse_val_for_bad_data)
        foo_da_tmp = np.log10(foo_da_tmp)
        foo_da_tmp = foo_da_tmp.sel(trial_type=trial_type)
        max_val = np.nanmax(foo_da_tmp.values)
        min_val = np.nanmin(foo_da_tmp.values)
        scalp_plot(
                ts,
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
        dirnm = os.path.basename(os.path.normpath(cfg_dataset["root_dir"]))
        if 'chromo' in foo_da.dims:
            title_str = f"{dirnm} - {name_conc_od} {trial_type} {foo_da.chromo.values[i_wav_chromo]} ({lower} to {upper} s)"
        else:
            title_str = f"{dirnm} - {name_conc_od} {trial_type} {foo_da.wavelength.values[i_wav_chromo]:.0f}nm ({lower} to {upper} s)"
        plt.suptitle(title_str)

        save_dir = os.path.join(cfg_dataset["root_dir"], 'derivatives', 'cedalion', cfg_dataset["derivatives_subfolder"], 'plots', 'DQR', 'group_weighted_avg')
        os.makedirs(save_dir, exist_ok=True)
        
        if 'chromo' in foo_da.dims:
            plt.savefig( os.path.join(save_dir, f'DQR_group_weighted_avg_{name_conc_od}_{trial_type}_{foo_da.chromo.values[i_wav_chromo]}.png') )
        else:
            plt.savefig( os.path.join(save_dir, f'DQR_group_weighted_avg_{name_conc_od}_{trial_type}_{foo_da.wavelength.values[i_wav_chromo]:.0f}nm.png') )
        plt.close()


def plot_mse_hist(trial_type, cfg_dataset, hrf_est_mse_subj, mse_val_for_bad_data, mse_min_thresh):
    # plot the MSE histogram
    ########################################################

    hrf_est_mse_subj_t = hrf_est_mse_subj #.sel(trial_type = trial_type)
    
    f,ax = plt.subplots(2,1,figsize=(6,10))

    # plot the diagonals for all subjects
    ax1 = ax[0]
    if 'time' in hrf_est_mse_subj_t.dims:
        foo = hrf_est_mse_subj_t.mean('time')
    else:
        foo = hrf_est_mse_subj_t
    
    if 'chromo' in hrf_est_mse_subj.dims:
        foo = foo.stack(measurement=['channel','chromo']).sortby('chromo')
        name_conc_od = 'conc'
    else:
        foo = foo.stack(measurement=['channel','wavelength']).sortby('wavelength')
        name_conc_od = 'od'

    n_subjects = foo.shape[0]  
    foo = foo.sel(trial_type=trial_type)  # select current trial type

    ax1.semilogy(foo.values.T, linewidth=0.5, alpha=0.5)  # all subjects at once
    ax1.set_title('variance in the mean for all subjects')
    ax1.set_xlabel('channel')
    # ax1.legend(loc="upper right")

    # histogram the diagonals
    ax1 = ax[1]
    foo1 = np.concatenate([foo[i] for i in range(n_subjects)]) # FIXME: need to loop over files too   # was foo[i][0] not sure what [0] was for, maybe trial type?
    # check if mse_val_for_bad_data has units
    if 'chromo' in hrf_est_mse_subj.dims:
        foo1 = np.where(foo1 == 0, mse_val_for_bad_data.magnitude, foo1) # some bad data gets through. amp=1e-6, but it is missed by the check above. Only 2 channels in 9 subjects. Seems to be channel 271
    else:
        foo1 = np.where(foo1 == 0, mse_val_for_bad_data, foo1) # remove 0s
    ax1.hist(np.log10(foo1), bins=100)
    
    if 'chromo' in hrf_est_mse_subj.dims:
        ax1.axvline(np.log10(mse_min_thresh.magnitude), color='r', linestyle='--', label=f'cov_min_thresh={mse_min_thresh.magnitude:.2e}')
    else:
        ax1.axvline(np.log10(mse_min_thresh), color='r', linestyle='--', label=f'cov_min_thresh={mse_min_thresh:.2e}')
        
    ax1.legend(loc="upper right")
    ax1.set_title(f'{name_conc_od} {trial_type} - histogram for all subjects of variance in the mean')
    ax1.set_xlabel('log10(cov_diag)')

    # give a title to the figure and save it
    dirnm = os.path.basename(os.path.normpath(cfg_dataset["root_dir"]))
    plt.suptitle(f'Data set - {dirnm}')

    save_dir = os.path.join(cfg_dataset["root_dir"], 'derivatives', 'cedalion', cfg_dataset["derivatives_subfolder"], 'plots', 'DQR', 'group_weighted_avg')
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig( os.path.join(save_dir, f'DQR_group_mse_histogram_{name_conc_od}_{trial_type}.png') )
    plt.close()



#%%

def main():
    
    cfg_dataset = snakemake.params.cfg_dataset
    cfg_groupaverage = snakemake.params.cfg_groupaverage
    cfg_hrf = snakemake.params.cfg_hrf
    
    hrf_files = snakemake.input.hrf_subs  
    
    out = snakemake.output[0]

    groupaverage_func(cfg_dataset, cfg_groupaverage, cfg_hrf, hrf_files, out)
    
            
if __name__ == "__main__":
    main()


