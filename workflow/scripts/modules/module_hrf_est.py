# module GLM
# functions to perform GLM

#%% Imports
import cedalion
import cedalion.nirs
import cedalion.sigproc.frequency as frequency
import cedalion.models.glm as glm
import cedalion.dataclasses as cdc
import xarray as xr
from cedalion import units
import numpy as np
import pandas as pd
from functools import reduce
import operator

#%% Functions

def blockaverage(epochs_all, cfg_hrf_estimation):
    all_trial_blockaverage = None
    spatial_dim = cdc.get_spatial_dimension(epochs_all)  # 'channel' or 'parcel'
    # Block Average
    baseline = epochs_all.sel(reltime=(epochs_all.reltime < 0)).mean('reltime')
    epochs = epochs_all - baseline  # baseline subtract
    blockaverage_ep = epochs.groupby('trial_type').mean('epoch') # mean across all epochs 

    epochs_zeromean = epochs - blockaverage_ep   # zeromean the epochs - residual
    
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
            epochs_zeromean_tmp = epochs_zeromean_tmp.stack(measurement=[spatial_dim,'chromo']).sortby('chromo')
            blockaverage = blockaverage.transpose('trial_type', spatial_dim, 'chromo', 'reltime')
        else:
            epochs_zeromean_tmp = epochs_zeromean_tmp.stack(measurement=[spatial_dim,'wavelength']).sortby('wavelength')
            blockaverage = blockaverage.transpose('trial_type', spatial_dim, 'wavelength', 'reltime')

        n_epochs = len(epochs_zeromean_tmp.epoch)

        epochs_zeromean_tmp = epochs_zeromean_tmp.transpose('trial_type', 'measurement', 'reltime', 'epoch')
        # calc mse
        mse_t = (epochs_zeromean_tmp**2).sum('epoch') / (n_epochs - 1)**2 # this is squared to get variance of the mean, aka MSE of the mean

        # retrieve channels/parcels where mse_t = 0
        bad_mask = mse_t.sel(trial_type=trial_type).data == 0
        bad_any = bad_mask.any(axis=1)
        bad_chans_mse = mse_t[spatial_dim][bad_any].values
    
        bad_chans_mse_lst.append(bad_chans_mse)

        if 'measurement' in mse_t.dims and isinstance(mse_t.get_index('measurement'), pd.MultiIndex):
            mse_t = mse_t.unstack('measurement')
        
        if all_trial_blockaverage is None:
            all_trial_blockaverage = blockaverage
            all_trial_mse = mse_t
        else:
            all_trial_blockaverage = xr.concat([all_trial_blockaverage, blockaverage], dim='trial_type') 
            all_trial_mse = xr.concat([all_trial_mse, mse_t], dim='trial_type')

    # DONE LOOP OVER TRIAL_TYPES

    return all_trial_blockaverage, all_trial_mse, bad_chans_mse_lst


def GLM(runs, cfg_hrf_estimation, geo3d, pruned_chans_list, short_sep_runs=None, posterior_var=None):
    """Fit the HRF GLM.

    ``runs`` is the data being modeled -- channel-space (rec_str = 'od'/'conc') for
    the default pipeline order, or parcel-space image-recon output (rec_str = 'conc',
    always molar) for the reconfirst pipeline order. Either way it's a list of either
    cedalion Records or (ts, stim) tuples -- see concatenate_runs().

    ``short_sep_runs``, if given, is a separate list of channel-space Records used
    only to compute the short-separation-channel nuisance regressor when
    cfg_GLM['do_short_sep'] is enabled. Short-separation regression needs real
    source/detector geometry, which parcel-space data doesn't have, so when ``runs``
    itself is parcel-space, the caller must supply the original channel-space
    preprocessed runs here instead; this regressor is then a single shared covariate
    applied uniformly across all parcels, not a per-parcel geometric regressor.
    Defaults to ``runs`` (today's channel-space behavior, unchanged).

    ``posterior_var``, if given, is the parcel-space Bayesian posterior variance of
    the reconstructed input Y (dims: spatial_dim, chromo; dequantified, same
    magnitude scale as Y), averaged across runs by the caller. When supplied, the
    GLM's beta covariance is inflated by ``weight = (var_resid + posterior_var) /
    var_resid`` before being projected into HRF-MSE, to account for the fact that Y
    itself is an uncertain, reconstructed quantity rather than a direct measurement
    (only meaningful for the reconfirst pipeline order, where image recon runs before
    the GLM). When ``None`` (default, channel-space GLM), no correction is applied
    and the corrected HRF-MSE return value is ``None``.

    ``cfg_GLM['do_global_signal']``, if true, adds a single nuisance regressor equal
    to the mean of ``Y_all`` across its spatial dimension (parcels, for the reconfirst
    pipeline order) at every timepoint, via
    ``cedalion.models.glm.design_matrix.global_mean_regressor``. Defaults to false.
    """
    cfg_GLM = cfg_hrf_estimation['GLM']
    rec_str = cfg_hrf_estimation['rec_str']

    # 1. need to concatenate runs
    Y_all, stim_df_tmp, runs_updated = concatenate_runs(runs, rec_str)

    target_units = Y_all.pint.units # grab units from data 
    target_units_time = str(Y_all.time.attrs['units'])

    # grab only trial type of interest
    stim_df = stim_df_tmp[stim_df_tmp['trial_type'].isin(cfg_hrf_estimation['stim_lst'])].reset_index(drop=True)
    
    # basis function options
    BASIS_FUNCTIONS = {
    "gaussian_kernels": glm.GaussianKernels,
    #"gaussian_kernels_with_tails": glm.GaussianKernelsWithTails, #FIXME: not currently in glm code?
    "gamma": glm.Gamma,
    # "gamma_deriv": glm.GammaDeriv, #FIXME: not currently in glm code?
    # "afni_gamma": glm.AFNIGamma, #FIXME: not currently in glm code?
    "dirac_delta": glm.DiracDelta, 
    }
    
    basis_cls = BASIS_FUNCTIONS[cfg_GLM["basis_func"]] # basis func class
    if basis_cls is None:
        raise ValueError(f"Unsupported basis function: {cfg_GLM['basis_func']}")
    if cfg_GLM["basis_func_params"] is None:
        raise ValueError(f"Missing parameters for basis function: {cfg_GLM['basis_func']}")

    if cfg_GLM['basis_func'] == "gaussian_kernels_with_tails" or cfg_GLM['basis_func'] == "gaussian_kernels":
        t_pre = cfg_hrf_estimation['t_pre']
        t_post = cfg_hrf_estimation['t_post']
    basis_params = cfg_GLM["basis_func_params"] # basis func params
    basis_func = basis_cls(t_pre=t_pre, t_post=t_post, **basis_params)  # basis function with params

    # 2. define design matrix
    dms = glm.design_matrix.hrf_regressors(
                                    Y_all,
                                    stim_df,
                                    basis_func
                                )

    # Combine drift and short-separation regressors (if any)
    if cfg_GLM['do_drift'] == 'polynomial': 
        drift_regressors = get_drift_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors)

    elif cfg_GLM['do_drift'] == 'legendre':  
        drift_regressors = get_drift_legendre_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors) # adds iteratively to dm 

    if cfg_GLM['do_short_sep']:
        if short_sep_runs is not None:
            # Parcel-space Y_all has no source/detector geometry for split_long_short_channels,
            # so build the short-sep regressor from the channel-space preprocessed runs instead,
            # concatenated with the same per-run time offsets as Y_all.
            # _, _, short_sep_runs_updated = concatenate_runs(short_sep_runs, 'od')
            _, _, short_sep_runs_updated = concatenate_runs(short_sep_runs, rec_str)
        else:
            short_sep_runs_updated = runs_updated
        ss_regressors = get_short_regressors(short_sep_runs_updated, pruned_chans_list, geo3d, cfg_GLM)
        dms &= reduce(operator.and_, ss_regressors)

    if cfg_GLM.get('do_global_signal', False):
        dms &= glm.design_matrix.global_mean_regressor(Y_all)

    dms.common = dms.common.fillna(0)

    # 3. get betas and covariance
    results = glm.fit(Y_all, dms, noise_model=cfg_GLM['noise_model'])  # fit GLM to get betas and covariance
    betas = results.sm.params  # this is the beta estimates for each regressor in the design matrix, it has dimensions regressor and measurement,
    cov_params = results.sm.cov_params() # this is the covariance of the beta estimates, which we can use to get MSE of the HRF estimate. It has dimensions regressor_r and regressor_c, ctions

    # 3b. reweight the beta covariance by the image-recon posterior variance, if given
    # (dequantify Y first so this is a plain-magnitude computation, matching how
    # posterior_var is stored -- see docstring)
    if posterior_var is not None:
        Y_plain = Y_all.pint.dequantify()
        posterior_var_plain = posterior_var.pint.dequantify() if hasattr(posterior_var, 'pint') else posterior_var
        resid = Y_plain - xr.dot(dms.common, betas, dim='regressor') # residuals of the GLM fit, dims: time, spatial_dim, chromo
        var_resid = resid.var('time')  # time-varying variance of the residuals, dims: spatial_dim, chromo
        weight = (var_resid + posterior_var_plain) / var_resid  # >= 1 elementwise

    # 4. estimate HRF and MSE
    basis_hrf = basis_func(Y_all)

    trial_type_list = cfg_hrf_estimation['stim_lst']

    hrf_mse_list = []
    hrf_mse_corrected_list = []
    hrf_estimate_list = []
    bad_chans_mse_lst = []

    for trial_type in trial_type_list:
        print(trial_type)
        betas_hrf = betas.sel(regressor=betas.regressor.str.startswith(f"HRF {trial_type}"))
        hrf_estimate = estimate_HRF_from_beta(betas_hrf, basis_hrf)
        
        cov_hrf = cov_params.sel(regressor_r=cov_params.regressor_r.str.startswith(f"HRF {trial_type}"),
                            regressor_c=cov_params.regressor_c.str.startswith(f"HRF {trial_type}") 
                                    )
        hrf_mse = estimate_HRF_cov(cov_hrf, basis_hrf)

        if posterior_var is not None:
            cov_hrf_reweighted = weight * cov_hrf
            hrf_mse_corrected = estimate_HRF_cov(cov_hrf_reweighted, basis_hrf)
            hrf_mse_corrected_list.append(hrf_mse_corrected.expand_dims({'trial_type': [trial_type]}))

        # get bad mse channels/parcels
        spatial_dim = cdc.get_spatial_dimension(hrf_mse)
        if 'chromo' in hrf_mse.dims:
            bad_mask = (hrf_mse == 0).any(dim=["time", "chromo"])
        else:
            bad_mask = (hrf_mse == 0).any(dim=["time", "wavelength"])
        bad_chans_mse = hrf_mse[spatial_dim][bad_mask].values

        hrf_estimate = hrf_estimate.expand_dims({'trial_type': [ trial_type ] })
        hrf_mse = hrf_mse.expand_dims({'trial_type': [ trial_type ] })

        hrf_estimate_list.append(hrf_estimate)
        hrf_mse_list.append(hrf_mse)
        bad_chans_mse_lst.append(bad_chans_mse)

    hrf_estimate = xr.concat(hrf_estimate_list, dim='trial_type')
    hrf_estimate = hrf_estimate.pint.quantify(target_units)

    hrf_mse = xr.concat(hrf_mse_list, dim='trial_type')
    hrf_mse = hrf_mse.pint.quantify(target_units**2)

    if posterior_var is not None:
        hrf_mse_corrected = xr.concat(hrf_mse_corrected_list, dim='trial_type')
        hrf_mse_corrected = hrf_mse_corrected.pint.quantify(target_units**2)
    else:
        hrf_mse_corrected = None

    # set universal time so that all hrfs have the same time base
    # (runs_updated is always a plain DataArray regardless of whether the caller
    # passed cedalion Records or (ts, stim) tuples -- see concatenate_runs())
    fs = frequency.sampling_rate(runs_updated[0]).to('Hz')
    before_samples = int(np.ceil((cfg_hrf_estimation['t_pre'] * fs).magnitude))
    after_samples = int(np.ceil((cfg_hrf_estimation['t_post'] * fs).magnitude))

    dT = np.round(1 / fs, 3)  # millisecond precision
    n_timepoints = len(hrf_estimate.time)
    reltime = np.linspace(-before_samples * dT, after_samples * dT, n_timepoints)

    hrf_mse = hrf_mse.assign_coords({'time': reltime})
    hrf_mse.time.attrs['units'] = target_units_time

    if hrf_mse_corrected is not None:
        hrf_mse_corrected = hrf_mse_corrected.assign_coords({'time': reltime})
        hrf_mse_corrected.time.attrs['units'] = target_units_time

    hrf_estimate = hrf_estimate.assign_coords({'time': reltime})
    hrf_estimate.time.attrs['units'] = target_units_time

    return results, hrf_estimate, hrf_mse, hrf_mse_corrected, bad_chans_mse_lst


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
        ts_long, ts_short = cedalion.nirs.split_long_short_channels(
                                rec_pruned, geo3d, distance_threshold= cfg_GLM['distance_threshold']  # !!! change to rec_pruned once NaN prob fixed
                                )
        # SSR options
        SSR_method = {
        "average": glm.design_matrix.average_short_channel_regressor(ts_short),
        "mean":  glm.design_matrix.average_short_channel_regressor(ts_short),
        "closest": glm.design_matrix.closest_short_channel_regressor(ts_long, ts_short, geo3d), 
        #'max_corr': glm.design_matrix.max_corr_short_channel_regressor(ts_long, ts_short), #FIXME: fails with NaNs
        }

        short = SSR_method[cfg_GLM['short_channel_method']]
        short.common = short.common.reset_coords('samples', drop=True)
        short.common = short.common.assign_coords({'regressor': [f'short run {i}']})
        ss_regressors.append(short)
        i = i+1

    return ss_regressors

def concatenate_runs(runs, rec_str):
    """Concatenate per-run time series into one continuous timeline.

    Each element of ``runs`` is either a cedalion Record (channel-space runs, loaded
    from a preprocessed snirf) or a plain (ts, stim) tuple (parcel-space runs, loaded
    from an image-recon .nc file, which has no Record to carry ``rec_str``/``.stim``).
    """

    CURRENT_OFFSET = 0
    runs_updated = []
    stim_updated = []

    for run in runs:

        if isinstance(run, tuple):
            ts, stim = run
        else:
            rec = run
            ts = rec[rec_str]
            stim = rec.stim
        
        units_attr = ts.time.attrs.get('units') # grab units

        time = ts.time.values
        new_time = time + CURRENT_OFFSET

        ts_new = ts.copy(deep=True)
        if rec_str == 'conc':
            ts_new = ts_new.pint.to('molar')
        ts_new = ts_new.assign_coords(time=new_time)

        if units_attr is not None:
            ts_new.time.attrs['units'] = units_attr # reassign units

        stim_shift = stim.copy()
        stim_shift['onset'] += CURRENT_OFFSET

        stim_updated.append(stim_shift)
        runs_updated.append(ts_new)

        CURRENT_OFFSET = new_time[-1] + (time[1] - time[0])  # updating time offset

    Y_all = xr.concat(runs_updated, dim='time')
    Y_all.time.attrs['units'] = ts.time.units 
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



