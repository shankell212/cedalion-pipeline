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

def img_recon_func(cfg_dataset, cfg_img_recon, cfg_hrf, groupaverage_path, out):
        
    # Load in data
    if os.path.exists(groupaverage_path):
        with open(groupaverage_path, 'rb') as f:
            groupavg_results = pickle.load(f)
      
        blockaverage_mean = groupavg_results['group_blockaverage_weighted']  #groupavg_results['group_blockaverage_weighted']
        blockaverage = groupavg_results['group_blockaverage']
        blockaverage_stderr = groupavg_results['total_stderr_blockaverage']
        blockaverage_subj = groupavg_results['blockaverage_subj']
        blockaverage_mse_subj = groupavg_results['blockaverage_mse_subj']
        geo2d = groupavg_results['geo2d']
        geo3d = groupavg_results['geo3d']  # !!! this is not in groupaverage results yet
        print(f" {groupaverage_path} loaded successfully!")
    
    else:
        print(f"Error: File '{groupaverage_path}' not found!")
                
    
    # Convert str vals to units from config
    cfg_sb = cfg_img_recon['spatial_basis']
    cfg_sb["threshold_brain"] = units(cfg_sb["threshold_brain"])
    cfg_sb["threshold_scalp"] = units(cfg_sb["threshold_scalp"])
    cfg_sb["sigma_brain"] = units(cfg_sb["sigma_brain"])
    cfg_sb["sigma_scalp"] = units(cfg_sb["sigma_scalp"])
    if isinstance(cfg_img_recon["mse_min_thresh"], str):
        cfg_img_recon["mse_min_thresh"] = float(cfg_img_recon["mse_min_thresh"])
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
    Adot = Adot.sel(channel=blockaverage_subj.channel.values)  # grab correct channels from probe

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
    threshold = -2 # log10 absolute  # !!! this is hard coded, add to config????
    wl_idx = 1

    ind_subj_blockavg = blockaverage_subj
    ind_subj_mse = blockaverage_mse_subj

    F = None
    D = None
    G = None

    all_trial_X_hrf_mag = None
    
    for trial_type in ind_subj_blockavg.trial_type:
        
        print(f'Getting images for trial type = {trial_type.values}')
        all_subj_X_hrf_mag = None
        
        for subj in ind_subj_blockavg.subj:
            print(f'Calculating subject = {subj.values}')
            #pdb.set_trace()
            od_hrf = ind_subj_blockavg.sel(subj=subj, trial_type=trial_type) 
            # od_hrf = od_hrf.stack(measurement=('channel', 'wavelength')).sortby('wavelength')
           
            od_mse = ind_subj_mse.sel(subj=subj, trial_type=trial_type).drop_vars(['subj', 'trial_type'])
            
            od_hrf_mag = od_hrf.sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime')
            od_mse_mag = od_mse.sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime')
            
            C_meas = od_mse_mag.pint.dequantify()
            C_meas = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength')
            C_meas = xr.where(C_meas < cfg_img_recon['mse_min_thresh'], cfg_img_recon['mse_min_thresh'], C_meas)

            # !!! GET RID of hard coded wavelength  -- can just grab from one of the xarrays?
            X_hrf_mag, W, D, F, G = img_recon.do_image_recon(od_hrf_mag, head = head, Adot = Adot, C_meas_flag = cfg_img_recon['Cmeas']['enable'], 
                                                             C_meas = C_meas, wavelength = [ind_subj_blockavg.wavelength[0].item(), ind_subj_blockavg.wavelength[1].item()], 
                                                             BRAIN_ONLY = cfg_img_recon['BRAIN_ONLY']['enable'], 
                                                             DIRECT = cfg_img_recon['DIRECT']['enable'], SB = cfg_sb['enable'], 
                                                        cfg_sbf = cfg_sb, alpha_spatial = cfg_img_recon['alpha_spatial'], 
                                                        alpha_meas = cfg_img_recon['alpha_meas'],F = F, D = D, G = G)
                                                        
            #pdb.set_trace()
            X_mse = img_recon.get_image_noise(C_meas, X_hrf_mag, W, DIRECT = cfg_img_recon['DIRECT']['enable'], SB= cfg_sb['enable'], G=G)
            

            # weighted average -- same as chan space - but now is vertex space  #pdb.set_trace()
            if all_subj_X_hrf_mag is None:
                
                all_subj_X_hrf_mag = X_hrf_mag
                all_subj_X_hrf_mag = all_subj_X_hrf_mag.assign_coords(subj=subj)
                all_subj_X_hrf_mag = all_subj_X_hrf_mag.assign_coords(trial_type=trial_type)

                all_subj_X_mse = X_mse
                all_subj_X_mse = all_subj_X_mse.assign_coords(subj=subj)
                all_subj_X_mse = all_subj_X_mse.assign_coords(trial_type=trial_type)

                X_hrf_mag_weighted = X_hrf_mag / X_mse
                X_mse_inv_weighted = 1 / X_mse   # X_mse = mse for 1 subject across all vertices , inverse is wt
                
            else:

                X_hrf_mag_tmp = X_hrf_mag.assign_coords(subj=subj)
                X_hrf_mag_tmp = X_hrf_mag_tmp.assign_coords(trial_type=trial_type)

                X_mse_tmp = X_mse.assign_coords(subj=subj)
                X_mse_tmp = X_mse_tmp.assign_coords(trial_type=trial_type)

                all_subj_X_hrf_mag = xr.concat([all_subj_X_hrf_mag, X_hrf_mag_tmp], dim='subj')
                all_subj_X_mse = xr.concat([all_subj_X_mse, X_mse_tmp], dim='subj')

                X_hrf_mag_weighted = X_hrf_mag_weighted + X_hrf_mag_tmp / X_mse
                X_mse_inv_weighted = X_mse_inv_weighted + 1 / X_mse       # summing weight over all subjects -- viz X_mse_inv_weighted will tell us which regions of brain we are most conf in
            # END OF SUBJECT LOOP

        # get the average
        X_hrf_mag_mean = all_subj_X_hrf_mag.mean('subj')
        
        X_hrf_mag_mean_weighted = X_hrf_mag_weighted / X_mse_inv_weighted
        
        X_mse_mean_within_subject = 1 / X_mse_inv_weighted
        X_mse_mean_within_subject = X_mse_mean_within_subject.assign_coords({'trial_type': trial_type})
            
        X_mse_weighted_between_subjects_tmp = (all_subj_X_hrf_mag - X_hrf_mag_mean_weighted)**2  
        X_mse_weighted_between_subjects = X_mse_weighted_between_subjects_tmp / all_subj_X_mse  
        X_mse_weighted_between_subjects = X_mse_weighted_between_subjects.mean('subj') * X_mse_mean_within_subject # normalized by the within subject variances as weights
     
        X_mse_weighted_between_subjects = X_mse_weighted_between_subjects.pint.dequantify()
     
        X_mse_btw_within_sum_subj = all_subj_X_mse + X_mse_weighted_between_subjects
        denom = (1/X_mse_btw_within_sum_subj).sum('subj')
        
        X_hrf_mag_mean_weighted = (all_subj_X_hrf_mag / X_mse_btw_within_sum_subj).sum('subj')  
        X_hrf_mag_mean_weighted = X_hrf_mag_mean_weighted / denom
        
        mse_total = 1/denom

        X_stderr_weighted = np.sqrt( mse_total )
        X_tstat = X_hrf_mag_mean_weighted / X_stderr_weighted
       
        if all_trial_X_hrf_mag is None:
            
            all_trial_X_hrf_mag = X_hrf_mag_mean
            all_trial_X_hrf_mag_weighted = X_hrf_mag_mean_weighted
            all_trial_X_stderr = X_stderr_weighted
            all_trial_X_tstat = X_tstat
            all_trial_X_mse_between = X_mse_weighted_between_subjects
            all_trial_X_mse_within = X_mse_mean_within_subject
        else:

            all_trial_X_hrf_mag = xr.concat([all_trial_X_hrf_mag, X_hrf_mag_mean], dim='trial_type')
            all_trial_X_hrf_mag_weighted = xr.concat([all_trial_X_hrf_mag_weighted, X_hrf_mag_mean_weighted], dim='trial_type')
            all_trial_X_stderr = xr.concat([all_trial_X_stderr, X_stderr_weighted], dim='trial_type')
            all_trial_X_tstat = xr.concat([all_trial_X_tstat, X_tstat], dim='trial_type')
            all_trial_X_mse_between = xr.concat([all_trial_X_mse_between, X_mse_weighted_between_subjects], dim='trial_type')
            all_trial_X_mse_within = xr.concat([all_trial_X_mse_within, X_mse_mean_within_subject], dim='trial_type')

    # END OF TRIAL TYPE LOOP
    results = {'X_hrf_mag': all_trial_X_hrf_mag,
               'X_hrf_mag_weighted': all_trial_X_hrf_mag_weighted,
               'X_std_err': all_trial_X_stderr,
               'X_tstat': all_trial_X_tstat,
               'X_mse_between': all_trial_X_mse_between,
               'X_mse_within': all_trial_X_mse_within
               }
    
    # Save data to a compressed pickle file 
    print(f'   Saving to {out}')
    file = gzip.GzipFile(out, 'wb')
    file.write(pickle.dumps(results))
    file.close()     
    
    
    #%% build and save plots
    if cfg_img_recon['plot_image']['enable']:
        plot_img = cfg_img_recon['plot_image']
        threshold = -2 # log10 absolute
        wl_idx = 1
        M = sbf.get_sensitivity_mask(Adot, threshold, wl_idx)
        flag_hbo_list = plot_img['flag_hbo_list']  #[True, False]
        flag_brain_list = plot_img['flag_brain_list'] #[True]   #, False]
        flag_img_list = plot_img['flag_img_list'] #['mag', 'tstat', 'noise'] #, 'noise'
            
        flag_condition_list = cfg_hrf['stim_lst']
        
        
        # all_trial_X_hrf_mag = results['X_hrf_mag']
        for flag_hbo in flag_hbo_list:
            
            for flag_brain in flag_brain_list: 
                
                for flag_condition in flag_condition_list:
                    
                    for flag_img in flag_img_list:
                        
                        if flag_hbo in ['hbo', 'HbO']:
                            title_str = flag_condition + ' ' + 'HbO'
                            hbx_brain_scalp = 'hbo'
                        else:
                            title_str = flag_condition + ' ' + 'HbR'
                            hbx_brain_scalp = 'hbr'
                        
                        if flag_brain in ['brain', 'Brain']:
                            title_str = title_str + ' brain'
                            hbx_brain_scalp = hbx_brain_scalp + '_brain'
                        else:
                            title_str = title_str + ' scalp'
                            hbx_brain_scalp = hbx_brain_scalp + '_scalp'
                        
                        if len(flag_condition_list) > 1:
                            if flag_img == 'tstat':
                                foo_img = all_trial_X_tstat.sel(trial_type=flag_condition).copy()
                                title_str = title_str + ' t-stat'
                            elif flag_img == 'mag':
                                foo_img = all_trial_X_hrf_mag_weighted.sel(trial_type=flag_condition).copy()
                                title_str = title_str + ' magnitude'
                            elif flag_img == 'noise':
                                foo_img = all_trial_X_stderr.sel(trial_type=flag_condition).copy()
                                title_str = title_str + ' noise'
                        else:
                            if flag_img == 'tstat':
                                foo_img = all_trial_X_tstat.copy()
                                title_str = title_str + ' t-stat'
                            elif flag_img == 'mag':
                                foo_img = all_trial_X_hrf_mag_weighted.copy()
                                title_str = title_str + ' magnitude'
                            elif flag_img == 'noise':
                                foo_img = all_trial_X_stderr.copy()
                                title_str = title_str + ' noise'
                
                        foo_img = foo_img.pint.dequantify()
                        foo_img = foo_img.transpose('vertex', 'chromo')
                        foo_img[~M] = np.nan
                        
                     # 
                        clim = (-foo_img.sel(chromo='HbO').max(), foo_img.sel(chromo='HbO').max())
                        # if flag_img == 'mag':
                        #     clim = [-7.6e-4, 7.6e-4]
                        p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (1,1), clim, hbx_brain_scalp, 'scale_bar',
                                                  None, title_str)
                        p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (0,0), clim, hbx_brain_scalp, 'left', p0)
                        p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (0,1), clim, hbx_brain_scalp, 'superior', p0)
                        p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (0,2), clim, hbx_brain_scalp, 'right', p0)
                        p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (1,0), clim, hbx_brain_scalp, 'anterior', p0)
                        p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (1,2), clim, hbx_brain_scalp, 'posterior', p0)
                        
    
                        #img_folder = f'{direct_name}_aspatial-{cfg_img_recon["alpha_spatial"]}_ameas-{cfg_img_recon["alpha_meas"]}_{Cmeas_name}_{SB_name}'
                        
                        img_folder = (
                                    #f"{cfg_dataset['root_dir']}/derivatives/{cfg_dataset['derivatives_subfolder']}/plots/image_recon/"
                                    ("direct" if cfg_img_recon["DIRECT"]["enable"] else "indirect") 
                                    + f"_aspatial-{cfg_img_recon['alpha_meas']}"
                                    + f"_ameas-{cfg_img_recon['alpha_spatial']}"
                                    + ("_Cmeas" if cfg_img_recon["Cmeas"]["enable"] else "_noCmeas")
                                    + ("_SB" if cfg_img_recon["spatial_basis"]["enable"] else "_noSB")
                                    )
                        #pdb.set_trace()
                        save_dir_tmp= os.path.join(cfg_dataset["root_dir"], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'image_recon', img_folder)
                        if not os.path.exists(save_dir_tmp):
                            os.makedirs(save_dir_tmp)
                        file_name = f'IMG_{flag_condition}_{flag_img}_{hbx_brain_scalp}.png'
                        p0.screenshot( os.path.join(save_dir_tmp, file_name) )
                        p0.close()
                        
                        
                     
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
    
    groupaverage_path = snakemake.input[0]
    
    out = snakemake.output[0]
    
    img_recon_func(cfg_dataset, cfg_img_recon, cfg_hrf, groupaverage_path, out)
    
            
if __name__ == "__main__":
    main()
