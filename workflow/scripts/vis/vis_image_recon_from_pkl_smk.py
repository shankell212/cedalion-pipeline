# %% Imports
##############################################################################
#%matplotlib widget

import os
import cedalion
import cedalion.nirs
import cedalion.xrutils as xrutils

import xarray as xr
import cedalion.plots as plots
from cedalion import units
import numpy as np

import gzip
import pickle
import json

import cedalion.dataclasses as cdc
import cedalion.datasets
import cedalion.geometry.registration
import cedalion.geometry.segmentation
import cedalion.imagereco.forward_model as fw
import cedalion.imagereco.tissue_properties
import cedalion.io
import cedalion.plots
import cedalion.sigproc.quality as quality
import cedalion.vis.plot_sensitivity_matrix
from cedalion import units
from cedalion.imagereco.solver import pseudo_inverse_stacked
from cedalion.io.forward_model import FluenceFile, load_Adot

scc = 0 if os.getenv("HOSTNAME") is None else 1

import sys
if scc == 1:
    # sys.path.append('/projectnb/nphfnirs/s/users/shannon/Code/cedalion-dab-funcs2/modules')
    sys.path.append('/projectnb/nphfnirs/s/users/shannon/Code/cedalion-pipeline/workflow/scripts/modules')
else:
    sys.path.append('C://Users//shank//Documents//GitHub//cedalion-dab-funcs2//modules')
import module_image_recon as img_recon 
import module_spatial_basis_funs as sbf    #module_spatial_basis_funs_ced

# Turn off all warnings
import warnings
warnings.filterwarnings('ignore')

#%%

ROOT_DIR = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS/" 
DERIV_DIR = os.path.join(ROOT_DIR, 'derivatives', 'Shannon', 'cedalion') #, 'test')

if scc ==1:
    #probe_dir = "/projectnb/nphfnirs/s/Shannon/Data/probes/NN22_WHHD/12NN/"  # CHANGE
    probe_dir = os.path.join(ROOT_DIR, 'derivatives/Shannon/cedalion/probe/')
else:
    probe_dir = 'C://Users//shank//Downloads//probes//NN22_WHHD//12NN//'
    
    
head_model = 'ICBM152'
snirf_name= 'fullhead_56x144_NN22_System1.snirf'

flag_condition_list = ['right', 'left']
SAVE = True
# folder_name = "Xs_BS_cov_alpha_spatial_1e-2_alpha_meas_1e4_indirect_Cmeas_SB" #"Xs_STS_cov_alpha_spatial_1e-3_alpha_meas_1e-2_indirect_Cmeas_noSB"  # CHANGE
folder_name = "Xs_BS_cov_alpha_spatial_1e-3_alpha_meas_1e4_indirect_Cmeas_noSB"#"Xs_BS_cov_alpha_spatial_1e-3_alpha_meas_1e4_indirect_Cmeas_noSB"

#%% Load head model 
import importlib
importlib.reload(img_recon)

head, PARCEL_DIR = img_recon.load_head_model(head_model, with_parcels=False)
Adot, meas_list, geo3d, amp = img_recon.load_probe(probe_dir, snirf_name = snirf_name) #snirf_name = "fullhead_56x144_System2.snirf")   #snirf_name='fullhead_56x144_NN22_System1.snirf')

ec = cedalion.nirs.get_extinction_coefficients('prahl', Adot.wavelength)
einv = cedalion.xrutils.pinv(ec)


#%% Open files

import importlib
importlib.reload(img_recon)

threshold = -2 # log10 absolute
wl_idx = 1
M = sbf.get_sensitivity_mask(Adot, threshold, wl_idx)
SAVE = True
flag_hbo_list = [True, False]
flag_brain_list = [True]   #, False]
flag_img_list = ['mag', 'tstat', 'noise'] #, 'noise'
    

if scc == 1:
    save_dir_tmp = os.path.join(DERIV_DIR, 'plots', 'image_recon')
    results_dir = os.path.join(DERIV_DIR, 'image_results')
    filepath = os.path.join(results_dir, f'{folder_name}.pkl.gz')
    with gzip.open( filepath, 'rb') as f:
        results = pickle.load(f)
else:
    save_dir_tmp = os.path.join("C://Users","shank", "Downloads", "image_results", folder_name)
    filepath = os.path.join("C:\\Users","shank", "Downloads", folder_name,f"{folder_name}.pkl")
    with open( filepath, 'rb') as f:
         results = pickle.load(f)

print("Files loaded successfully")

all_trial_X_hrf_mag = results['X_hrf_mag']
all_trial_X_hrf_mag_weighted = results['X_hrf_mag_weighted']
all_trial_X_stderr = results['X_std_err']
all_trial_X_tstat = results['X_tstat'] 
all_trial_X_mse_between = results['X_mse_between'] 
all_trial_X_mse_within = results['X_mse_within'] 
               

#%% Plot
import importlib
importlib.reload(img_recon)

# all_trial_X_hrf_mag = results['X_hrf_mag']
for flag_hbo in flag_hbo_list:
    
    for flag_brain in flag_brain_list: 
        
        for flag_condition in flag_condition_list:
            
            for flag_img in flag_img_list:
                
                if flag_hbo:
                    title_str = flag_condition + ' ' + 'HbO'
                    hbx_brain_scalp = 'hbo'
                else:
                    title_str = flag_condition + ' ' + 'HbR'
                    hbx_brain_scalp = 'hbr'
                
                if flag_brain:
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
                                          None, title_str, off_screen=SAVE )
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (0,0), clim, hbx_brain_scalp, 'left', p0)
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (0,1), clim, hbx_brain_scalp, 'superior', p0)
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (0,2), clim, hbx_brain_scalp, 'right', p0)
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (1,0), clim, hbx_brain_scalp, 'anterior', p0)
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (1,2), clim, hbx_brain_scalp, 'posterior', p0)
                
                
                if SAVE:
                    save_dir_tmp_ful = os.path.join(save_dir_tmp, folder_name)
                    if not os.path.exists(save_dir_tmp_ful):
                        os.makedirs(save_dir_tmp_ful)
                    file_name = f'IMG_{flag_condition}_{flag_img}_{hbx_brain_scalp}.png'
                    p0.screenshot( os.path.join(save_dir_tmp_ful, file_name) )
                    p0.close()
                    print(f"Images saved to {save_dir_tmp_ful}")
                else:
                    p0.show()
                
                # if SAVE:
                #     img_folder = f'{direct_name}_aspatial-{cfg_img_recon["alpha_spatial"]}_ameas-{cfg_img_recon["alpha_meas"]}_{Cmeas_name}_{SB_name}'
                #     save_dir_tmp= os.path.join(cfg_dataset["root_dir"], 'derivatives', cfg_dataset['derivatives_subfolder'], 'plots', 'image_recon', img_folder)
                #     if not os.path.exists(save_dir_tmp):
                #         os.makedirs(save_dir_tmp)
                #     file_name = f'IMG_{flag_condition}_{flag_img}_{hbx_brain_scalp}.png'
                #     p0.screenshot( os.path.join(save_dir_tmp, file_name) )
                #     p0.close()
                # else:
                #     p0.show()