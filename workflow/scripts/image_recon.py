#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 07:17:55 2025

@author: smkelley
"""


import os
import cedalion
import cedalion.nirs
import cedalion.xrutils as xrutils

from cedalion.physunits import units
import xarray as xr
import cedalion.plots as plots
import numpy as np

import gzip
import pickle
import json
import pdb

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.join(script_dir, 'modules')
sys.path.append(modules_path)

import module_image_recon as img_recon 
import module_spatial_basis_funs as sbf 
import pyvista as pv
pv.OFF_SCREEN = True

# Turn off all warnings
import warnings
warnings.filterwarnings('ignore')


#%%

def img_recon_func(cfg_dataset, cfg_img_recon, cfg_hrf, hrf_file, out):
                
    
    # Convert str vals to units from config
    cfg_mse = cfg_img_recon['mse']
    cfg_sb = cfg_img_recon['spatial_basis']
    cfg_sb["threshold_brain"] = units(cfg_sb["threshold_brain"])
    cfg_sb["threshold_scalp"] = units(cfg_sb["threshold_scalp"])
    cfg_sb["sigma_brain"] = units(cfg_sb["sigma_brain"])
    cfg_sb["sigma_scalp"] = units(cfg_sb["sigma_scalp"])
    if isinstance(cfg_mse["mse_min_thresh"], str):
        cfg_mse["mse_min_thresh"] = float(eval(cfg_mse["mse_min_thresh"]))
    if isinstance(cfg_mse["mse_amp_thresh"], str):
        cfg_mse["mse_amp_thresh"] = float(eval(cfg_mse["mse_amp_thresh"]))
    if isinstance(cfg_mse["hrf_val"], str):
                cfg_mse["hrf_val"] = float(cfg_mse["hrf_val"])
    if isinstance(cfg_mse["mse_val_for_bad_data"], str):
                cfg_mse["mse_val_for_bad_data"] = float(cfg_mse["mse_val_for_bad_data"])

    if isinstance(cfg_img_recon["alpha_meas"], str):
        cfg_img_recon["alpha_meas"] = float(cfg_img_recon["alpha_meas"])
    if isinstance(cfg_img_recon["alpha_spatial"], str):
        cfg_img_recon["alpha_spatial"] = float(cfg_img_recon["alpha_spatial"])

    
        
    #%% Load head model 
    head, PARCEL_DIR = img_recon.load_head_model(cfg_img_recon['head_model'], with_parcels=False)
    Adot, meas_list, geo3d, amp = img_recon.load_probe(cfg_img_recon['probe_dir'], snirf_name=cfg_img_recon['snirf_name_probe'])
    
    ec = cedalion.nirs.get_extinction_coefficients(cfg_img_recon['spectrum'], Adot.wavelength)
    einv = cedalion.xrutils.pinv(ec)
    
    #%% run image recon
    #pdb.set_trace()
    

    # # Make Adot and blockaverage channel order the same
    # blockaverage_subj = blockaverage_subj.sel(channel=Adot.channel.values)
    # blockaverage_mse_subj = blockaverage_mse_subj.sel(channel=Adot.channel.values)
    
    """
    do the image reconstruction of each subject independently 
    - this is the unweighted subject block average magnitude 
    - then reconstruct their individual MSE
    - then get the weighted average in image space 
    - get the total standard error using between + within subject MSE 
    """
    threshold = -2 # log10 absolute  # !!! this is hard coded, add to config????  # !!! mask_threshold
    wl_idx = 1

    F = None
    D = None
    G = None

    all_trial_X_hrf_mag = None
    
    for idxt, trial_type in enumerate(cfg_hrf['stim_lst']):
        
        print(f'Getting images for trial type = {trial_type}')

        # load in block average files
        with gzip.open(hrf_file, 'rb') as f:
            results = pickle.load(f)
        hrf_est = results['hrf_est']
        mse_t = results['mse_t'] 
        bad_channels = results['bad_indices']        

        hrf = hrf_est.sel(trial_type=trial_type) 
        mse = mse_t.sel(trial_type=trial_type).drop_vars(['trial_type'])

        # Convert conc to od and units for cfg
        if 'chromo' in hrf_est.dims:
            dpf = xr.DataArray(
                    [1, 1],
                    dims="wavelength",
                    coords={"wavelength": amp.wavelength},
                    )
            E = cedalion.nirs.get_extinction_coefficients('prahl', amp.wavelength)
            od_hrf =  cedalion.nirs.conc2od(hrf, geo3d, dpf)
            od_mse = xr.dot(E**2, mse, dim =['chromo']) * 1 * units.mm**2

        else:
            od_hrf = hrf.copy()
            od_mse = mse.copy()

        Adot = Adot.sel(channel=hrf_est.channel.values)  # grab correct channels from probe

        print(f'Calculating subject = {hrf_file}')

        # replace bad vals
        od_hrf.loc[dict(channel=bad_channels)] = cfg_mse['hrf_val']
        od_mse.loc[dict(channel=bad_channels)] = cfg_mse['mse_val_for_bad_data']
        od_mse = xr.where(od_mse < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], od_mse)  # !!! maybe can be removed when we have the between subject mse
        
        # if doing magnitude image
        if cfg_img_recon['mag']['enable']:
            if 'reltime' in od_hrf.dims:
                od_hrf_mag = od_hrf.sel(reltime=slice(cfg_img_recon['mag']['t_win'][0], cfg_img_recon['mag']['t_win'][1])).mean('reltime')
                od_mse_mag = od_mse.sel(reltime=slice(cfg_img_recon['mag']['t_win'][0], cfg_img_recon['mag']['t_win'][1])).mean('reltime')
            else:
                od_hrf_mag = od_hrf.sel(time=slice(cfg_img_recon['mag']['t_win'][0], cfg_img_recon['mag']['t_win'][1])).mean('time')
                od_mse_mag = od_mse.sel(time=slice(cfg_img_recon['mag']['t_win'][0], cfg_img_recon['mag']['t_win'][1])).mean('time')
        else:
            od_hrf_mag = od_hrf.copy()
            if 'reltime' in od_hrf.dims:
                od_mse_mag = od_mse.mean('reltime')
            else:
                 od_mse_mag.mean('time')

        C_meas = od_mse_mag.pint.dequantify()
        C_meas = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength')
        C_meas = xr.where(C_meas < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], C_meas)
        od_mse_ts = od_mse.stack(measurement=('channel', 'wavelength')).sortby('wavelength')
        #pdb.set_trace()
        X_hrf_mag, W, D, F, G = img_recon.do_image_recon(od_hrf_mag, head = head, Adot = Adot, C_meas_flag = cfg_img_recon['Cmeas']['enable'], 
                                                            C_meas = C_meas, wavelength = [od_hrf.wavelength[0].item(), od_hrf.wavelength[1].item()], 
                                                            BRAIN_ONLY = cfg_img_recon['BRAIN_ONLY']['enable'], 
                                                            DIRECT = cfg_img_recon['DIRECT']['enable'], SB = cfg_sb['enable'], 
                                                    cfg_sbf = cfg_sb, alpha_spatial = cfg_img_recon['alpha_spatial'], 
                                                    alpha_meas = cfg_img_recon['alpha_meas'],F = F, D = D, G = G)
        if 'reltime' in od_mse_ts.dims:                                            
            od_mse_ts = od_mse_ts.transpose('measurement', 'reltime')
        else:
             od_mse_ts = od_mse_ts.transpose('measurement', 'time')
             
        if cfg_img_recon['mag']['enable']:
            X_mse = img_recon.get_image_noise(od_mse_mag, X_hrf_mag, W, DIRECT = cfg_img_recon['DIRECT']['enable'], SB= cfg_sb['enable'], G=G)
        else:
            X_mse = img_recon.get_image_noise(od_mse, X_hrf_mag, W, DIRECT = cfg_img_recon['DIRECT']['enable'], SB= cfg_sb['enable'], G=G)
        
        # concatenate trial 
        X_hrf_mag = X_hrf_mag.assign_coords(trial_type=trial_type) # add trial type name as a coordinate
        X_mse = X_mse.assign_coords(trial_type=trial_type)

        if all_trial_X_hrf_mag is None:
            all_trial_X_hrf_mag = X_hrf_mag
            all_trial_X_mse = X_mse
        else:
            all_trial_X_hrf_mag = xr.concat([all_trial_X_hrf_mag, X_hrf_mag], dim='trial_type')
            all_trial_X_mse = xr.concat([all_trial_X_mse, X_mse], dim='trial_type')

    # END OF TRIAL TYPE LOOP
    results = { 'hrf_est': all_trial_X_hrf_mag,
                'mse_t': all_trial_X_mse,
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
    
    cfg_dataset = snakemake.params.cfg_dataset  # get params
    cfg_img_recon = snakemake.params.cfg_img_recon
    cfg_hrf = snakemake.params.cfg_hrf
    
    hrf_data = snakemake.input.hrf_data
    
    out = snakemake.output[0]
    
    img_recon_func(cfg_dataset, cfg_img_recon, cfg_hrf, hrf_data, out)
    
            
if __name__ == "__main__":
    main()
