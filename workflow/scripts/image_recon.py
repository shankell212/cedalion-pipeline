#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 07:17:55 2025

@author: smkelley
"""


import os
import cedalion
import cedalion.nirs

from cedalion.physunits import units
import xarray as xr
from cedalion.sigproc.quality import measurement_variance
import cedalion.dot as dot
import cedalion.io as io
import cedalion.vis as plots
import numpy as np
import gzip
import pickle
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.join(script_dir, 'modules')
sys.path.append(modules_path)

import module_image_recon as img_recon 
#import module_spatial_basis_funs as sbf 
import pyvista as pv
pv.OFF_SCREEN = True

# Turn off all warnings
import warnings
warnings.filterwarnings('ignore')


#%%

def img_recon_func(cfg_img_recon, cfg_hrf, file_name, Adot_path, geo_path, out, SB=[]):

    # Convert str vals to units from config
    cfg_mse = cfg_img_recon['mse']
    cfg_sb = cfg_img_recon['spatial_basis']
    cfg_sb["threshold_brain"] = units(cfg_sb["threshold_brain"])
    cfg_sb["threshold_scalp"] = units(cfg_sb["threshold_scalp"])
    cfg_sb["sigma_brain"] = units(cfg_sb["sigma_brain"])
    cfg_sb["sigma_scalp"] = units(cfg_sb["sigma_scalp"])
    if isinstance(cfg_mse["mse_min_thresh"], str):
        cfg_mse["mse_min_thresh"] = float(eval(cfg_mse["mse_min_thresh"]))
    if isinstance(cfg_mse["hrf_val"], str):
                cfg_mse["hrf_val"] = float(cfg_mse["hrf_val"])
    if isinstance(cfg_mse["mse_val_for_bad_data"], str):
                cfg_mse["mse_val_for_bad_data"] = float(cfg_mse["mse_val_for_bad_data"])

    if isinstance(cfg_img_recon["alpha_meas"], str):
        cfg_img_recon["alpha_meas"] = float(cfg_img_recon["alpha_meas"])
    if isinstance(cfg_img_recon["alpha_spatial"], str):
        cfg_img_recon["alpha_spatial"] = float(cfg_img_recon["alpha_spatial"])

    if isinstance(cfg_img_recon["lambda_spatial_depth"], str):
            cfg_img_recon["lambda_spatial_depth"] = float(eval(cfg_img_recon["lambda_spatial_depth"]))
    
        
    #%% Load head model and sensitivity matrix

    head = dot.get_standard_headmodel(cfg_img_recon['head_model'])
    Adot = io.forward_model.load_Adot(Adot_path)

    ec = cedalion.nirs.get_extinction_coefficients(cfg_img_recon['spectrum'], Adot.wavelength)
    
    # load geometry 
    with gzip.open(geo_path, 'rb') as f:
        geo_pos = pickle.load(f)
        geo3d = geo_pos['geo3d']

    #%% run image recon
    
    """
    do the image reconstruction of each subject independently 
    - this is the unweighted subject block average magnitude 
    - then reconstruct their individual MSE
    - then get the weighted average in image space 
    - get the total standard error using between + within subject MSE 
    """
    
    
    # load files
    if 'hrf' in file_name:
        with gzip.open(file_name, 'rb') as f:
            results = pickle.load(f)
        ts = results['hrf_est']
        mse_t = results['mse_t'] 
        bad_channels = results['bad_indices'] 

    elif 'preprocess' in file_name:
        with gzip.open(file_name, 'rb') as f:
            record = pickle.load(f)
        rec = record[0]
        ts = rec['od_corrected'].copy()
        mse_t = None
        # add bad_indices to file that also has rec in it

    # Loop through trial types
    all_trial_Xs = None
    for idxt, trial_type in enumerate(cfg_hrf['stim_lst']): #NOTE: do we still need to loop through trial types here?
        
        print(f'Getting images for trial type = {trial_type}')       

        ts_trial = ts.sel(trial_type=trial_type) 
        if mse_t is not None: # if hrf data loaded in
            mse_trial = mse_t.sel(trial_type=trial_type).drop_vars(['trial_type'])

        # Convert conc to od and units for cfg
        if 'chromo' in ts.dims:
            dpf = xr.DataArray(
                    [1, 1],
                    dims="wavelength",
                    coords={"wavelength": Adot.wavelength},
                    )
            od_ts =  cedalion.nirs.cw.conc2od(ts_trial, geo3d, dpf)
            if mse_t is not None:  # would not need this in theory bc if loading in ts, then conc should not be there 
                od_mse = xr.dot(ec**2, mse_trial, dim =['chromo']) * 1 * units.mm**2  

        else:
            od_ts = ts_trial.copy()
            if mse_t is not None: # if mse variable exists, i.e. loading in hrf not ts
                od_mse = mse_trial.copy()
            else:
                mse = measurement_variance(od_ts, calc_covariance=False) #NOTE: CHECK DIMS 
                od_mse = mse.sel(trial_type=trial_type).drop_vars(['trial_type'])

        print(f'Calculating subject = {file_name}')

        # replace bad vals
        od_ts.loc[dict(channel=bad_channels)] = cfg_mse['hrf_val']
        od_mse.loc[dict(channel=bad_channels)] = cfg_mse['mse_val_for_bad_data']
        od_mse = xr.where(od_mse < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], od_mse)  # !!! maybe can be removed when we have the between subject mse
        
        # if doing magnitude image
        if cfg_img_recon['mag']['enable']:
            if 'reltime' in od_ts.dims:
                od_ts_mag = od_ts.sel(reltime=slice(cfg_img_recon['mag']['t_win'][0], cfg_img_recon['mag']['t_win'][1])).mean('reltime')
                #od_mse_mag = od_mse.sel(reltime=slice(cfg_img_recon['mag']['t_win'][0], cfg_img_recon['mag']['t_win'][1])).mean('reltime')
            else:
                od_ts_mag = od_ts.sel(time=slice(cfg_img_recon['mag']['t_win'][0], cfg_img_recon['mag']['t_win'][1])).mean('time')
                #od_mse_mag = od_mse.sel(time=slice(cfg_img_recon['mag']['t_win'][0], cfg_img_recon['mag']['t_win'][1])).mean('time')
        else:
            od_ts_mag = od_ts.copy()

        if mse_t is not None: # if hrf loaded in, get mse magnitude
            if 'reltime' in od_ts.dims:
                od_mse_mag = od_mse.mean('reltime')
            else:
                od_mse_mag = od_mse.mean('time')
        else:
             od_mse_mag = od_mse.copy() # if mse not loaded in, copy od_mse

        C_meas = od_mse_mag.pint.dequantify()
        #C_meas = np.diag(C_meas)

        #C_meas = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength') #NOTE: do we need to do this anymore? check shape of c_meas from output func
        
        # save G (spatial basis) in derivatives/cedalion/forward_model  -> for brain and scalp separately and sigma
       
        if cfg_sb['enable'] and SB:  # do I need both
            #fil_path, after = Adot_path.split("fw", 1)
            print('Performing image recon with SB')
            with gzip.open(SB, 'rb') as f:
                sbf = pickle.load(f)

            recon = dot.ImageRecon(
                    Adot,
                    recon_mode=cfg_img_recon['recon_mode'],
                    brain_only = cfg_img_recon['BRAIN_ONLY']['enable'],
                    alpha_meas = cfg_img_recon['alpha_meas'],
                    alpha_spatial = cfg_img_recon['alpha_spatial'],
                    apply_c_meas = cfg_img_recon['Cmeas']['enable'],
                    spatial_basis_functions = sbf,
                )
        else:
             sbf = None
             print('Performing image recon without SB')
             recon = dot.ImageRecon(
                    Adot,
                    recon_mode=cfg_img_recon['recon_mode'],  # conc is direct, mua2conc is indirect
                    brain_only = cfg_img_recon['BRAIN_ONLY']['enable'],
                    alpha_meas = cfg_img_recon['alpha_meas'],
                    alpha_spatial = cfg_img_recon['alpha_spatial'],
                    apply_c_meas = cfg_img_recon['Cmeas']['enable'],
                    spatial_basis_functions = None,
                )

        if cfg_img_recon['Cmeas']['enable']:
             Xs = recon.reconstruct(od_ts_mag, C_meas)
        else:
             Xs = recon.reconstruct(od_ts_mag)
        
        
        #X_mse = recon.get_image_noise(C_meas) # get image noise   #NOTE: how to handle noise computation in pipeline workflow when not using Cmeas?
        if cfg_img_recon['recon_mode']=='conc':
             DIRECT = True
        elif cfg_img_recon['recon_mode'] == 'mua2conc':
             DIRECT = False
            
        X_mse = img_recon.get_image_noise_posterior(Adot, C_meas, alpha_meas = cfg_img_recon['alpha_meas'], 
                                                    alpha_spatial_depth = cfg_img_recon['alpha_meas'], 
                                                lambda_spatial_depth =  cfg_img_recon['lambda_spatial_depth'], 
                                                DIRECT=DIRECT, SB=cfg_img_recon['spatial_basis']['enable'], G=sbf)       

        # concatenate trial 
        Xs = Xs.assign_coords(trial_type=trial_type) # add trial type name as a coordinate
        X_mse = X_mse.assign_coords(trial_type=trial_type)
        
        if all_trial_Xs is None:
            all_trial_Xs = Xs.expand_dims(trial_type=[trial_type])
            all_trial_X_mse = X_mse.expand_dims(trial_type=[trial_type])
        else:
            all_trial_Xs = xr.concat([all_trial_Xs, Xs], dim='trial_type')
            all_trial_X_mse = xr.concat([all_trial_X_mse, X_mse], dim='trial_type')

    # IF loading in time series, save in parcel space instead of vertex for smaller file size
    if mse_t is None: # if time series data loaded in
         Xs_parcel_weighted = (
                    (all_trial_Xs / all_trial_X_mse)            # numerator weights: X * (1/var)
                    .groupby("parcel")
                    .sum("vertex")
                    /
                    (1 / all_trial_X_mse)
                    .groupby("parcel")
                    .sum("vertex")
                )

    # END OF TRIAL TYPE LOOP
    if mse_t is not None:
        results = { 'Xs': all_trial_Xs,  #NOTE: change name later
                    'mse': all_trial_X_mse,
                }
    else:
         results = { 'Xs': Xs_parcel_weighted,  #NOTE: change name later
                                    }
    
    # Save data to a compressed pickle file 
    print(f'   Saving to {out}')
    file = gzip.GzipFile(out, 'wb')
    file.write(pickle.dumps(results))
    file.close()     
    
    
    #%% build and save plots
    # if cfg_img_recon['plot_image']['enable']:
    #     plot_img = cfg_img_recon['plot_image']
    #     threshold = -2 # log10 absolute
    #     wl_idx = 1
    #     M = sbf.get_sensitivity_mask(Adot, threshold, wl_idx)
    #     flag_hbo_list = plot_img['flag_hbo_list']  #[True, False]
    #     flag_brain_list = plot_img['flag_brain_list'] #[True]   #, False]
    #     flag_img_list = plot_img['flag_img_list'] #['mag', 'tstat', 'noise'] #, 'noise'
            
    #     flag_condition_list = cfg_hrf['stim_lst']
        
        
    #     # all_trial_X_hrf_mag = results['X_hrf_mag']
    #     for flag_hbo in flag_hbo_list:
            
    #         for flag_brain in flag_brain_list: 
                
    #             for flag_condition in flag_condition_list:
                    
    #                 for flag_img in flag_img_list:
                        
    #                     if flag_hbo in ['hbo', 'HbO']:
    #                         title_str = flag_condition + ' ' + 'HbO'
    #                         hbx_brain_scalp = 'hbo'
    #                     else:
    #                         title_str = flag_condition + ' ' + 'HbR'
    #                         hbx_brain_scalp = 'hbr'
                        
    #                     if flag_brain in ['brain', 'Brain']:
    #                         title_str = title_str + ' brain'
    #                         hbx_brain_scalp = hbx_brain_scalp + '_brain'
    #                     else:
    #                         title_str = title_str + ' scalp'
    #                         hbx_brain_scalp = hbx_brain_scalp + '_scalp'
                        
    #                     if len(flag_condition_list) > 1:
    #                         if flag_img == 'tstat':
    #                             foo_img = all_trial_X_tstat.sel(trial_type=flag_condition).copy()
    #                             title_str = title_str + ' t-stat'
    #                         elif flag_img == 'mag':
    #                             foo_img = all_trial_X_hrf_mag_weighted.sel(trial_type=flag_condition).copy()
    #                             title_str = title_str + ' magnitude'
    #                         elif flag_img == 'noise':
    #                             foo_img = all_trial_X_stderr.sel(trial_type=flag_condition).copy()
    #                             title_str = title_str + ' noise'
    #                     else:
    #                         if flag_img == 'tstat':
    #                             foo_img = all_trial_X_tstat.copy()
    #                             title_str = title_str + ' t-stat'
    #                         elif flag_img == 'mag':
    #                             foo_img = all_trial_X_hrf_mag_weighted.copy()
    #                             title_str = title_str + ' magnitude'
    #                         elif flag_img == 'noise':
    #                             foo_img = all_trial_X_stderr.copy()
    #                             title_str = title_str + ' noise'
                
    #                     foo_img = foo_img.pint.dequantify()
    #                     foo_img = foo_img.transpose('vertex', 'chromo')
    #                     foo_img[~M] = np.nan
                        
    #                  # 
    #                     clim = (-foo_img.sel(chromo='HbO').max(), foo_img.sel(chromo='HbO').max())
    #                     # if flag_img == 'mag':
    #                     #     clim = [-7.6e-4, 7.6e-4]
    #                     p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (1,1), clim, hbx_brain_scalp, 'scale_bar',
    #                                               None, title_str)
    #                     p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (0,0), clim, hbx_brain_scalp, 'left', p0)
    #                     p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (0,1), clim, hbx_brain_scalp, 'superior', p0)
    #                     p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (0,2), clim, hbx_brain_scalp, 'right', p0)
    #                     p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (1,0), clim, hbx_brain_scalp, 'anterior', p0)
    #                     p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (1,2), clim, hbx_brain_scalp, 'posterior', p0)
                        
    
    #                     #img_folder = f'{direct_name}_aspatial-{cfg_img_recon["alpha_spatial"]}_ameas-{cfg_img_recon["alpha_meas"]}_{Cmeas_name}_{SB_name}'
                        
    #                     img_folder = (
    #                                 #f"{cfg_dataset['root_dir']}/derivatives/{cfg_dataset['derivatives_subfolder']}/plots/image_recon/"
    #                                 ("direct" if cfg_img_recon["DIRECT"]["enable"] else "indirect") 
    #                                 + f"_aspatial-{cfg_img_recon['alpha_meas']}"
    #                                 + f"_ameas-{cfg_img_recon['alpha_spatial']}"
    #                                 + ("_Cmeas" if cfg_img_recon["Cmeas"]["enable"] else "_noCmeas")
    #                                 + ("_SB" if cfg_img_recon["spatial_basis"]["enable"] else "_noSB")
    #                                 )
    #                     #pdb.set_trace()
    #                     save_dir_tmp= os.path.join(cfg_dataset["root_dir"], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'image_recon', img_folder)
    #                     if not os.path.exists(save_dir_tmp):
    #                         os.makedirs(save_dir_tmp)
    #                     file_name = f'IMG_{flag_condition}_{flag_img}_{hbx_brain_scalp}.png'
    #                     p0.screenshot( os.path.join(save_dir_tmp, file_name) )
    #                     p0.close()
                        
                        
                     
#%%

# SAVING plots dummy code 

# recon_plot_folder = (
#             f"{ROOT}/derivatives/{DERIV}/plots/image_recon/"
#             +("_direct" if config["image_recon"]["DIRECT"]["enable"] else "_indirect") 
#             + f"_aspatial-{config['image_recon']['alpha_meas']}"
#             + f"_ameas-{config['image_recon']['alpha_spatial']}"
#             + ("_Cmeas" if config["image_recon"]["Cmeas"]["enable"] else "_noCmeas")
#             + ("_SB" if config["image_recon"]["spatial_basis"]["enable"] else "_noSB")
#             )

# expand(recon_plot_folder + 'IMG_{flag_condition}_{flag_img}_{hbx}_{brain_scalp}.png', 
#        flag_img = config['image_recon']['plot_image']['flag_img_list'], 
#        flag_condition = config['hrf']['stim_lst'], 
#        brain_scalp = config['image_recon']['plot_image']['flag_brain_list'],
#        hbx = config['image_recon']['plot_image']['flag_hbo_list'], 
#        )  
            
            
#         #     + f"_cov_alpha_spatial_{config['image_recon']['alpha_spatial']}"
#         #     + f"_alpha_meas_{config['image_recon']['alpha_meas']}"
#         #     + ("_direct" if config["image_recon"]["DIRECT"]["enable"] else "_indirect")
#         #     + ("_Cmeas" if config["image_recon"]["Cmeas"]["enable"] else "_noCmeas")
#         #     + ("_SB" if config["image_recon"]["spatial_basis"]["enable"] else "_noSB")
#         #     + ".pkl.gz"
#         # )
    

    


#%%

def main():
    config = snakemake.config
    
    # get params
    cfg_img_recon = snakemake.params.cfg_img_recon
    cfg_hrf = snakemake.params.cfg_hrf
    
    hrf_data = snakemake.input.hrf_data
    geo_path = snakemake.input.geometry
    Adot_path = snakemake.input.Adot
    SB_path = snakemake.input.SB

    Adot_path = str(Adot_path) if not isinstance(Adot_path, str) else Adot_path
    
    out = snakemake.output[0]
    
    img_recon_func(cfg_img_recon, cfg_hrf, hrf_data, Adot_path, geo_path, out, SB_path)
    
            
if __name__ == "__main__":
    main()


