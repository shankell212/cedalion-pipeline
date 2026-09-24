#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot brain maps (magnitude / t-stat / noise) from an IMAGE-SPACE group-average
result -- i.e. the output of the default Snakefile's `groupaverage_imgspace`
rule, or Snakefile_reconfirst's `groupaverage` rule (both produce a
Xs_groupavg_*.nc file with vertex geometry).

This is a standalone script, NOT wired into the Snakemake DAG -- just point
GROUPAVG_NC at an existing group-average .nc file and run it directly:

    python plot_groupavg_brainmap.py

It renders one image per (condition x HbO/HbR x brain/scalp x mag/tstat/noise)
combination, using the same cedalion image_recon_multi_view() rendering that
image_recon.py uses for per-subject plots.

NOTE: the plain channel-space group average (task-*_nirs_groupaverage_chanspace_*.nc,
from the default Snakefile's `groupaverage` rule) has no vertex dimension --
it's channel-space HRF data, not a reconstructed image, and can't be plotted
this way. Only Xs_groupavg_*.nc (image-space) has brain geometry to render.

Adapted from workflow/scripts/vis/vis_image_recon_from_pkl_smk_new.py (the
legacy per-pkl group-average plotting script for the old BU-cluster pipeline),
updated to read the current .nc output format and current sensitivity-matrix
loading convention (see image_recon.py).

@author: shank
"""

import os
import numpy as np
import xarray as xr
import cedalion
import cedalion.nirs
import cedalion.io as io
from cedalion.vis.anatomy import image_recon_multi_view

import pyvista as pv
pv.OFF_SCREEN = True

import warnings
warnings.filterwarnings('ignore')

#%% ---- EDIT THESE ----

# Path to the IMAGE-SPACE group-average result (Xs_groupavg_*.nc).
GROUPAVG_NC = (
    "/Users/shannonkelley/Documents/fNIRS/Data/test_data_cedalion_smk/data/"
    "derivatives/cedalion/test_norm/Outputs/group_results/"
    "Xs_groupavg_BS_cov_alpha_spatial_1e-3_alpha_meas_1e4_recon_mode_mua2conc_Cmeas_noSB_mag_5_8.nc"
)

# Sensitivity matrix (Adot) used to build the sensitivity mask -- same one the
# image recon that produced GROUPAVG_NC used.
ADOT_PATH = (
    "/Users/shannonkelley/Documents/fNIRS/Data/test_data_cedalion_smk/data/"
    "derivatives/cedalion/forward/shannon/sensitivity.nc"
)

HEAD_MODEL = 'icbm152'

# Trial types / conditions to plot. Leave as None to auto-detect from the
# file's trial_type coordinate; override with an explicit list (e.g.
# ['left', 'right']) if you only want a subset.
FLAG_CONDITION_LIST = None

FLAG_HBO_LIST   = ['hbo', 'hbr']             # HbO / HbR
FLAG_BRAIN_LIST = ['brain', 'scalp']         # brain surface / scalp surface
FLAG_IMG_LIST   = ['mag', 'tstat', 'noise']  # which stat to render

# Where to save the rendered images. Defaults to a 'plots/group_results/<name>'
# folder next to Outputs/group_results/.
SAVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(GROUPAVG_NC)), 'plots', 'group_results'
)

#%% ---- Helpers ----

def quantify_if_needed(da):
    """Restore pint units from da.attrs['units'] if present (groupaverage.py
    dequantifies before saving to netcdf, per the codebase's usual
    'dequant to save, re-quant on load' convention). Falls back to the plain
    array if there's nothing to quantify (e.g. tstat may be unitless)."""
    try:
        return da.pint.quantify()
    except Exception as e:
        print(f"  (leaving '{da.name}' unquantified: {e})")
        return da

#%% ---- Load ----

Adot = io.forward_model.load_Adot(ADOT_PATH)
head = cedalion.dot.get_standard_headmodel(HEAD_MODEL)

ds = xr.open_dataset(GROUPAVG_NC)

if 'vertex' not in ds['group_average_weighted'].dims:
    raise ValueError(
        f"{GROUPAVG_NC} has no 'vertex' dimension -- this looks like a "
        "channel-space group average (from the plain `groupaverage` rule), "
        "not an image-space one. Point GROUPAVG_NC at a Xs_groupavg_*.nc file "
        "(from `groupaverage_imgspace`) instead."
    )

all_trial_groupaverage_weighted = quantify_if_needed(ds['group_average_weighted'])
all_trial_X_stderr = quantify_if_needed(ds['total_stderr'])
all_trial_X_tstat = quantify_if_needed(ds['tstat'])

condition_list = FLAG_CONDITION_LIST or list(ds['trial_type'].values)

#%% ---- Restrict to sensitive vertices (same convention as image_recon.py) ----

intensity = np.log10(Adot[:, :, 0].sum('channel'))
mask = intensity > -2
sensitivity_mask = mask.drop_vars('wavelength')

all_trial_groupaverage_weighted = all_trial_groupaverage_weighted.where(sensitivity_mask)
all_trial_X_stderr = all_trial_X_stderr.where(sensitivity_mask)
all_trial_X_tstat = all_trial_X_tstat.where(sensitivity_mask)

#%% ---- Plot ----

folder_name = os.path.basename(GROUPAVG_NC).removesuffix('.nc')

for flag_hbo in FLAG_HBO_LIST:
    for flag_brain in FLAG_BRAIN_LIST:
        for flag_condition in condition_list:
            for flag_img in FLAG_IMG_LIST:

                if flag_hbo in ('hbo', 'HbO'):
                    title_str = f'{flag_condition} HbO'
                    hbx_brain_scalp = 'hbo'
                else:
                    title_str = f'{flag_condition} HbR'
                    hbx_brain_scalp = 'hbr'

                if flag_brain in ('brain', 'Brain'):
                    title_str += ' brain'
                    hbx_brain_scalp += '_brain'
                else:
                    title_str += ' scalp'
                    hbx_brain_scalp += '_scalp'

                if flag_img == 'tstat':
                    foo_img = all_trial_X_tstat.sel(trial_type=flag_condition).copy()
                    title_str += ' t-stat'
                elif flag_img == 'mag':
                    foo_img = all_trial_groupaverage_weighted.sel(trial_type=flag_condition).copy()
                    title_str += ' magnitude'
                elif flag_img == 'noise':
                    foo_img = all_trial_X_stderr.sel(trial_type=flag_condition).copy()
                    title_str += ' noise'
                else:
                    raise ValueError(f"Unknown flag_img: {flag_img}")

                if 'reltime' in foo_img.dims:
                    foo_img = foo_img.rename({'reltime': 'time'})
                    foo_img = foo_img.transpose('vertex', 'chromo', 'time')

                clim = (-foo_img.sel(chromo='HbO').max(), foo_img.sel(chromo='HbO').max())

                filename = f'IMG_groupavg_{flag_condition}_{flag_img}_{hbx_brain_scalp}'
                save_dir_full = os.path.join(SAVE_DIR, folder_name)
                os.makedirs(save_dir_full, exist_ok=True)
                save_file_path = os.path.join(save_dir_full, filename)

                print('plotting: ', filename)
                image_recon_multi_view(
                    foo_img,
                    head,
                    cmap='jet',
                    clim=clim,
                    view_type=hbx_brain_scalp,
                    title_str=f'{filename} / uM',
                    filename=save_file_path,
                    SAVE=True,
                    fps=12,
                    geo3d_plot=None,
                    wdw_size=(1024, 768),
                )

print(f"Done. Plots saved under: {SAVE_DIR}")
