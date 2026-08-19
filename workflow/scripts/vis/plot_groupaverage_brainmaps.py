#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_groupaverage_brainmaps.py

Render brain-surface (parcel-space) plots of the reconfirst-pipeline group-average
HRF result (Outputs/group_results/task-{task}_nirs_groupaverage_imgspace_hrf_*.nc).

WHY THIS SCRIPT EXISTS
-----------------------
Neither Snakefile_reconfirst nor groupaverage.py currently generates brain plots for
the reconfirst pipeline:
  - image_recon.py's built-in plotting only fires when 'trial_type' is present in the
    reconstructed data (see the `if cfg_img_recon['plot_image']['enable'] and
    'trial_type' in all_trial_Xs.dims:` block). In the reconfirst order, image recon
    runs on the continuous per-run time series *before* HRF estimation/epoching, so
    that dim never exists there and the plot block is always skipped.
  - groupaverage.py's own plotting functions (plot_mean_stderr, plot_mse_hist) are
    commented out (see the `# FIXME: group DQR plots` block).

This script fills that gap for the group-average step. It:
  1. Loads the group-average .nc file (parcel-space: dims trial_type, parcel, chromo,
     time).
  2. Loads the same standard head model used during image recon (cedalion ships it
     with per-vertex parcel labels already assigned -- see
     cedalion.dot.get_standard_headmodel's docstring).
  3. Broadcasts each parcel's value out to every brain vertex belonging to that
     parcel, since cedalion's brain-plotting functions (image_recon_multi_view) only
     know how to color a vertex-resolution mesh, not a parcel-resolution one.
  4. Calls cedalion.vis.anatomy.image_recon_multi_view (the same function
     image_recon.py's per-run plotting already uses) to render + save PNG/GIF plots.

RUN THIS LOCALLY
-----------------
This needs the `cedalion_smk` conda environment (cedalion, xarray, pint, pyvista,
matplotlib) -- run it on your machine, e.g.:

    conda activate cedalion_smk
    python plot_groupaverage_brainmaps.py

Edit the CONFIG block below if your paths/params differ.
"""

import os
import numpy as np
import xarray as xr

import cedalion
import cedalion.dot
import cedalion.io as io
from cedalion.vis.anatomy import image_recon_multi_view

import pyvista as pv
pv.OFF_SCREEN = True

import warnings
warnings.filterwarnings('ignore')


# %% ------------------------------------------------------------------------------
# CONFIG -- edit these if your setup differs from Snakefile_reconfirst.yaml
# -----------------------------------------------------------------------------

ROOT_DIR = "/Users/shannonkelley/Documents/fNIRS/Data/test_data_cedalion_smk/data"
DERIV_SUBFOLDER = "test_new_ced_reconfirst"
TASK = "BS"
HEAD_MODEL = "icbm152"          # image_recon.generate_sensitivity.head_model in config

# The forward-model sensitivity matrix (same file used by every subject's per-run
# reconstruction -- generate_sensitivity_matrix.py builds ONE Adot from a single
# representative subject's probe geometry and reuses it for the whole study, since
# everyone shares the same montage). Needed here purely for masking: without it,
# parcels/vertices your probe doesn't actually see still get plotted, because the
# regularized image-recon solve returns *some* estimate everywhere, not just where
# there's real sensitivity.
ADOT_SUB_FOLDER = "shannon"     # image_recon.generate_sensitivity.sub_folder in config
ADOT_FILE = os.path.join(
    ROOT_DIR, "derivatives", "cedalion", "forward", ADOT_SUB_FOLDER, "sensitivity.nc"
)
SENSITIVITY_LOG10_THRESH = -2   # same threshold image_recon.py's per-run plots use

# Filename suffix must match _hrf_imgspace_suffix() in Snakefile_reconfirst
# (rec_str + imagerecon params) for the file you actually have on disk.
GROUPAVG_SUFFIX = (
    "conc_imgHRF_parcelspace_cov_alpha_spatial_1e-3_alpha_meas_1e4"
    "_recon_mode_mua2conc_Cmeas_noSB"
)

GROUPAVG_FILE = os.path.join(
    ROOT_DIR, "derivatives", "cedalion", DERIV_SUBFOLDER, "Outputs", "group_results",
    f"task-{TASK}_nirs_groupaverage_imgspace_hrf_{GROUPAVG_SUFFIX}.nc",
)

SAVE_DIR = os.path.join(
    ROOT_DIR, "derivatives", "cedalion", DERIV_SUBFOLDER, "plots", "group_results",
    GROUPAVG_SUFFIX,
)

# Which result variable(s) to plot. group_average_weighted = weighted group-avg HRF
# magnitude; tstat = group_average_weighted / total_stderr. Both already live in the
# saved dataset, so no recomputation needed (unlike image_recon.py's per-run plots).
VARS_TO_PLOT = {
    "mag": "group_average_weighted",
    "tstat": "tstat",
}

CHROMO_LIST = ["HbO", "HbR"]          # which chromophores to render
TRIAL_TYPES = None                    # None = plot every trial_type found in the file

ANIMATE = False   # True -> one .gif per condition/chromo/var, animated across the
                   # full epoch (t_pre..t_post). False -> a single static .png at
                   # PEAK_TIME_S (faster; good for a first look).
PEAK_TIME_S = 5.0  # only used when ANIMATE = False; picks the nearest time sample


# %% ------------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------

print(f"Loading group-average file:\n  {GROUPAVG_FILE}")
ds = xr.open_dataset(GROUPAVG_FILE)

if TRIAL_TYPES is None:
    trial_types = list(ds["trial_type"].values)
else:
    trial_types = TRIAL_TYPES

print(f"Trial types found: {trial_types}")
print(f"Parcels: {ds.sizes.get('parcel')}, chromo: {list(ds['chromo'].values)}, "
      f"time samples: {ds.sizes.get('time')}")


# %% ------------------------------------------------------------------------------
# LOAD HEAD MODEL + BUILD PARCEL -> VERTEX BROADCAST MAP
# -----------------------------------------------------------------------------

print(f"Loading standard head model: {HEAD_MODEL}")
head = cedalion.dot.get_standard_headmodel(HEAD_MODEL)

if "parcel" not in head.brain.vertex_coords:
    raise RuntimeError(
        "head.brain.vertex_coords has no 'parcel' entry. The standard head model is "
        "documented to ship with parcel labels already assigned "
        "(cedalion.dot.get_standard_headmodel docstring), so this is unexpected -- "
        "check your cedalion version, or assign parcels yourself first via "
        "head.assign_parcels_via_mni_coords(...) before running this script."
    )

parcel_of_vertex = np.asarray(head.brain.vertex_coords["parcel"])
n_vertex = parcel_of_vertex.shape[0]
print(f"Head model has {n_vertex} brain vertices across "
      f"{len(np.unique(parcel_of_vertex))} distinct parcel labels.")

vertex_index = xr.DataArray(parcel_of_vertex, dims="vertex")


def parcel_to_vertex(da_parcel: xr.DataArray) -> xr.DataArray:
    """Broadcast a (..., parcel, ...) DataArray out to (..., vertex, ...).

    Every brain vertex takes the value of the parcel it belongs to. Vertices whose
    parcel label isn't present in da_parcel's 'parcel' coordinate (e.g. parcels with
    too little sensitivity to have been reconstructed) get NaN, which
    image_recon_multi_view already renders as light gray via nan_color.
    """
    all_labels = np.unique(parcel_of_vertex)
    da_aligned = da_parcel.reindex(parcel=all_labels, fill_value=np.nan)
    da_vertex = da_aligned.sel(parcel=vertex_index)
    da_vertex = da_vertex.assign_coords(
        is_brain=("vertex", np.ones(n_vertex, dtype=bool))
    )
    # image_recon() indexes with a plain boolean array on axis 0
    # (X.sel(chromo=...)[X.is_brain.values]), so 'vertex' must be the first dim.
    da_vertex = da_vertex.transpose("vertex", ...)
    return da_vertex


# %% ------------------------------------------------------------------------------
# LOAD Adot + BUILD SENSITIVITY MASK
# -----------------------------------------------------------------------------
# Same convention as image_recon.py's (already-working) per-run/per-subject plots:
# a vertex only counts as "seen" if the summed sensitivity across all channels (at
# the first wavelength) is above a floor. Below that floor the regularized inverse
# solve is essentially unconstrained there -- it still returns *some* estimate
# (driven by the spatial prior, not real data), and for a ratio quantity like tstat
# (mag / stderr) both the numerator and denominator can shrink together in those
# unconstrained regions, so the ratio can come out large even with zero real signal
# underneath it. Masking these out (-> NaN -> gray) is what keeps the per-run plots
# honest, and it was missing from the first version of this script.

print(f"Loading sensitivity matrix (Adot) for masking:\n  {ADOT_FILE}")
Adot = io.forward_model.load_Adot(ADOT_FILE)

# Adot's 'vertex' dim spans brain vertices then scalp vertices (see
# ForwardModel.compute_sensitivity); restrict to the brain subset via its own
# is_brain coordinate so it lines up with our brain-only parcel_of_vertex array,
# built from the same head model.
Adot_brain = Adot.sel(vertex=Adot.is_brain.values)
if Adot_brain.sizes["vertex"] != n_vertex:
    raise RuntimeError(
        f"Adot has {Adot_brain.sizes['vertex']} brain vertices but the head model "
        f"has {n_vertex}. Make sure ADOT_FILE and HEAD_MODEL both come from the "
        "same Snakefile_reconfirst.yaml run -- a mismatched forward model/head "
        "model pairing will silently misalign the sensitivity mask."
    )

intensity = np.log10(Adot_brain[:, :, 0].sum("channel"))
sensitivity_mask = (intensity > SENSITIVITY_LOG10_THRESH).drop_vars(
    "wavelength", errors="ignore"
)
n_sensed = int(sensitivity_mask.sum())
print(f"Sensitivity mask: {n_sensed}/{n_vertex} brain vertices pass the "
      f"log10 > {SENSITIVITY_LOG10_THRESH} threshold.")


# %% ------------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

for var_key, var_name in VARS_TO_PLOT.items():
    if var_name not in ds:
        print(f"Skipping '{var_name}' -- not found in dataset.")
        continue

    # reattach pint units the same way groupaverage.py does when it reloads these
    # files (they were dequantified before being written to disk)
    data = ds[var_name].pint.quantify()

    for trial_type in trial_types:
        data_tt = data.sel(trial_type=trial_type)

        # keep the 'chromo' dim intact -- image_recon() (called inside
        # image_recon_multi_view) does its own X.sel(chromo='HbO'/'HbR') internally
        # regardless of view_type, so it needs 'chromo' to still be a real dimension
        # on whatever we pass in, not pre-selected out.
        data_vertex = parcel_to_vertex(data_tt)
        data_vertex = data_vertex.where(sensitivity_mask)

        # pick the time frame (or full time range for animation) once, before the
        # chromo loop, so both chromo plots use the exact same underlying data
        if not ANIMATE:
            data_vertex = data_vertex.sel(time=PEAK_TIME_S, method="nearest")
            plotted_time = float(data_vertex.time)
        else:
            plotted_time = None

        for chromo in CHROMO_LIST:
            hbx = "hbo_brain" if chromo == "HbO" else "hbr_brain"
            title_str = f"{trial_type} {chromo} {var_key}"
            filename = f"GROUPAVG_{trial_type}_{var_key}_{hbx}"
            save_file_path = os.path.join(SAVE_DIR, filename)

            clim_max = float(
                np.nanmax(np.abs(
                    data_vertex.sel(chromo=chromo).pint.dequantify().values
                ))
            )
            # clim = (-clim_max, clim_max)
            clim = (-1.3e-6, 1.3e-6)  # hard-coded to match the per-run plots for now

            if ANIMATE:
                print(f"Rendering GIF: {filename}")
                image_recon_multi_view(
                    data_vertex,
                    head,
                    cmap="jet",
                    clim=clim,
                    view_type=hbx,
                    title_str=title_str,
                    filename=save_file_path,
                    SAVE=True,
                    fps=12,
                    geo3d_plot=None,
                    wdw_size=(1024, 768),
                )
            else:
                print(f"Rendering static frame ({plotted_time:.2f}s): {filename}")
                image_recon_multi_view(
                    data_vertex,
                    head,
                    cmap="jet",
                    clim=clim,
                    view_type=hbx,
                    title_str=title_str,
                    filename=save_file_path,
                    SAVE=True,
                    geo3d_plot=None,
                    wdw_size=(1024, 768),
                )

print(f"\nDone. Plots saved under:\n  {SAVE_DIR}")
