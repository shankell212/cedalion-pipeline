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
from cedalion.physunits import units
from cedalion.imagereco.solver import pseudo_inverse_stacked
from cedalion.io.forward_model import FluenceFile, load_Adot
from cedalion.plots import image_recon_multi_view

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

ROOT_DIR_probe = "/projectnb/nphfnirs/s/datasets/Interactive_Walking_HD/" #"/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS/" 
ROOT_DIR = "/projectnb/nphfnirs/s/users/shannon/Data/reg_test_data/test_data/"
DERIV_DIR = os.path.join(ROOT_DIR, 'derivatives', 'cedalion') #, 'test')

#probe_dir = "/projectnb/nphfnirs/s/Shannon/Data/probes/NN22_WHHD/12NN/"  # CHANGE
probe_dir = os.path.join(ROOT_DIR_probe, 'derivatives/cedalion/probe/')

    
head_model = 'ICBM152'
snirf_name= 'fullhead_56x144_NN22_System1.snirf'

task = 'BS'
flag_condition_list = ['right', 'left']
SAVE = True
# folder_name = "Xs_BS_cov_alpha_spatial_1e-2_alpha_meas_1e4_indirect_Cmeas_SB" #"Xs_STS_cov_alpha_spatial_1e-3_alpha_meas_1e-2_indirect_Cmeas_noSB"  # CHANGE
#folder_name = f"Xs_{task}_cov_alpha_spatial_1e-2_alpha_meas_1e4_indirect_Cmeas_SB_ts"

# folder_name = "task-BS_nirs_groupaverage_imgspace.pkl"
folder_name = "Xs_groupavg_BS_cov_alpha_spatial_1e-3_alpha_meas_1e4_indirect_Cmeas_noSB_mag.pkl"

#%% Load head model 
import importlib
importlib.reload(img_recon)

head, PARCEL_DIR = img_recon.load_head_model(head_model, with_parcels=False)
Adot, meas_list, geo3d, amp = img_recon.load_probe(probe_dir, snirf_name = snirf_name) #snirf_name = "fullhead_56x144_System2.snirf")   #snirf_name='fullhead_56x144_NN22_System1.snirf')

ec = cedalion.nirs.get_extinction_coefficients('prahl', Adot.wavelength)
einv = cedalion.xrutils.pinv(ec)


#%% Open files
    

save_dir_tmp = os.path.join(DERIV_DIR, 'plots', 'image_recon')
results_dir = os.path.join(DERIV_DIR, 'image_results')
filepath = os.path.join(results_dir, f'{folder_name}')
with open( filepath, 'rb') as f:
    results = pickle.load(f)

print("Files loaded successfully")


all_trial_groupaverage_weighted = results['group_average_weighted']
all_trial_X_stderr = results['total_stderr']
all_trial_X_tstat = results['tstat']



#%% Plot w cedalion function:

SAVE = True
flag_hbo_list = [True, False]
flag_brain_list = [True]   #, False]
flag_img_list = ['mag', 'tstat', 'noise'] #, 'noise'

# scl = np.percentile(np.abs(X_ts.sel(chromo='HbO').values.reshape(-1)),99)
# clim = (-scl,scl)

for flag_hbo in flag_hbo_list:
    
    for flag_brain in flag_brain_list: 
        
        for flag_condition in flag_condition_list:
            
            for flag_img in flag_img_list:

                if flag_hbo:
                    title_str = flag_condition + ' ' + 'HbO'
                    hbx_brain_scalp = 'hbo'
                    chromo='HbO'
                else:
                    title_str = flag_condition + ' ' + 'HbR'
                    hbx_brain_scalp = 'hbr'
                    chromo='HbR'
                
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
                        foo_img = all_trial_groupaverage_weighted.sel(trial_type=flag_condition).copy()
                        title_str = title_str + ' magnitude'
                    elif flag_img == 'noise':
                        foo_img = all_trial_X_stderr.sel(trial_type=flag_condition).copy()
                        title_str = title_str + ' noise'
                else:
                    if flag_img == 'tstat':
                        foo_img = all_trial_X_tstat.copy()
                        title_str = title_str + ' t-stat'
                    elif flag_img == 'mag':
                        foo_img = all_trial_groupaverage_weighted.copy()
                        title_str = title_str + ' magnitude'
                    elif flag_img == 'noise':
                        foo_img = all_trial_X_stderr.copy()
                        title_str = title_str + ' noise'
                
                if 'reltime' in foo_img.dims:
                    foo_img = foo_img.rename({"reltime": "time"})
                    foo_img = foo_img.transpose("vertex", "chromo", "time")
                # else:
                #     foo_img = foo_img.transpose("vertex", "chromo")

                #foo_img = foo_img.pint.dequantify()
                #arr = foo_img.sel(chromo=chromo).pint.to("micromolar").pint.magnitude
                #scl = np.percentile(np.abs(arr.sel(chromo=chromo).values.reshape(-1)),99)
                #scl = np.percentile(np.abs(arr.reshape(-1)),99)
                #clim = (-scl,scl)
                #clim = (-foo_img.sel(chromo=chromo).max(), foo_img.sel(chromo=chromo).max())

                arr = foo_img.sel(chromo=chromo).values
                arr = arr[np.isfinite(arr)]
                scl = np.max(np.abs(arr)) if arr.size > 0 else 1.0
                clim = (-scl, scl)

                if SAVE:
                    save_dir_tmp_ful = os.path.join(save_dir_tmp, folder_name)
                    if not os.path.exists(save_dir_tmp_ful):
                        os.makedirs(save_dir_tmp_ful)
                filename = f'IMG_{flag_condition}_{flag_img}_{hbx_brain_scalp}'
                save_file_path = os.path.join(save_dir_tmp_ful, filename )

                print(foo_img)
                print('plotting: ', filename)
                image_recon_multi_view(
                    foo_img,  # time series data; can be 2D (static) or 3D (dynamic)
                    head,
                    cmap='jet',
                    clim=clim,
                    view_type=hbx_brain_scalp,
                    title_str=f'{filename} / uM',
                    filename=save_file_path,
                    SAVE=SAVE,
                    #time_range=(foo_img.time.values[0],foo_img.time.values[-1],0.5)*units.s,
                    fps=6,
                    geo3d_plot = None, #  geo3d_plot
                    wdw_size = (1024, 768)
                )

#