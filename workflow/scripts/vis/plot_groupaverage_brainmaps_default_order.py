#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_groupaverage_brainmaps_default_order.py

Render brain-surface plots of the DEFAULT-pipeline-order group-average image-recon
result (derivatives_subfolder "test_order2", Outputs/group_results/Xs_groupavg_*.nc).

HOW THIS DIFFERS FROM plot_groupaverage_brainmaps.py (the reconfirst version)
------------------------------------------------------------------------------
The default Snakefile order is preprocess -> HRF estimation -> image recon ->
group average -- the reverse of Snakefile_reconfirst. That ordering changes the
shape of the group-average file in two ways that matter here:

  1. VERTEX-space, not parcel-space. image_recon.py reconstructs from an already-
     epoched HRF estimate (has 'trial_type'), so it keeps full vertex resolution
     (ds_results['Xs'] = all_trial_Xs, not the parcel-grouped mean used for the
     reconfirst pipeline's continuous per-run reconstruction). The saved group
     average already carries 'vertex' as its spatial dim, with 'parcel' and
     'is_brain' riding along as per-vertex coordinates baked in by
     cedalion.dot.image_recon's reconstruct() -- so unlike the reconfirst script,
     there's no parcel->vertex broadcasting step needed here at all.
  2. Static, not a time course. image_recon.mag.enable=true in this config, so
     image_recon.py reconstructs from a magnitude (peak-window) reduction of the
     HRF estimate rather than the full HRF curve -- the group-average file has no
     'time'/'reltime' dimension, just one value per (trial_type, vertex, chromo).
     So there's no animate-vs-static choice to make; every plot is a single frame.
  3. Both brain AND scalp views are meaningful here (image_recon.BRAIN_ONLY is
     false and the data spans both, unlike the reconfirst parcel-space data which
     is brain-only), so this script renders both.

What's unchanged from the reconfirst version: cedalion's brain-plotting functions
still color a real vertex mesh, so unsensed vertices (outside your probe's actual
coverage) still need masking by the forward-model sensitivity matrix (Adot) or
they'll get colored by whatever the regularized inverse solve's prior left there --
same issue, same fix, as the reconfirst script. Both pipeline orders share the
exact same Adot file (same probe montage, same 'shannon' forward-model folder).

RUN THIS LOCALLY
-----------------
Needs the `cedalion_smk` conda environment (cedalion, xarray, pint, pyvista,
matplotlib):

    conda activate cedalion_smk
    python plot_groupaverage_brainmaps_default_order.py

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
# CONFIG -- edit these if your setup differs from config_test_1.yml
# -----------------------------------------------------------------------------

ROOT_DIR = "/Users/shannonkelley/Documents/fNIRS/Data/test_data_cedalion_smk/data"
DERIV_SUBFOLDER = "test_order2"
TASK = "BS"
HEAD_MODEL = "icbm152"          # image_recon.generate_sensitivity.head_model in config

# Same forward-model sensitivity matrix used by both pipeline orders (one shared
# probe montage / 'shannon' subfolder) -- see plot_groupaverage_brainmaps.py for
# why this masking step matters.
ADOT_SUB_FOLDER = "shannon"
ADOT_FILE = os.path.join(
    ROOT_DIR, "derivatives", "cedalion", "forward", ADOT_SUB_FOLDER, "sensitivity.nc"
)
SENSITIVITY_LOG10_THRESH = -2   # same threshold image_recon.py's per-run plots use

# Filename suffix must match get_groupavg_imagerecon_output() in Snakefile for the
# file you actually have on disk (alpha_spatial/alpha_meas/recon_mode/Cmeas/SB/
# mag+t_win, all from config['image_recon']).
GROUPAVG_SUFFIX = (
    "cov_alpha_spatial_1e-3_alpha_meas_1e4_recon_mode_mua2conc_Cmeas_noSB_mag_5_8"
)

GROUPAVG_FILE = os.path.join(
    ROOT_DIR, "derivatives", "cedalion", DERIV_SUBFOLDER, "Outputs", "group_results",
    f"Xs_groupavg_{TASK}_{GROUPAVG_SUFFIX}.nc",
)

SAVE_DIR = os.path.join(
    ROOT_DIR, "derivatives", "cedalion", DERIV_SUBFOLDER, "plots", "group_results",
    GROUPAVG_SUFFIX,
)

# Which result variable(s) to plot -- all three already live in the saved dataset,
# no recomputation needed.
VARS_TO_PLOT = {
    "mag": "group_average_weighted",
    "tstat": "tstat",
    "noise": "total_stderr",
}

CHROMO_LIST = ["HbO", "HbR"]
VIEW_LIST = ["brain", "scalp"]         # this data spans both, unlike reconfirst's
TRIAL_TYPES = None                     # None = plot every trial_type found in the file


# %% ------------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------

print(f"Loading group-average file:\n  {GROUPAVG_FILE}")
ds = xr.open_dataset(GROUPAVG_FILE)

if TRIAL_TYPES is None:
    trial_types = list(ds["trial_type"].values)
else:
    trial_types = TRIAL_TYPES

n_vertex = ds.sizes.get("vertex")
print(f"Trial types found: {trial_types}")
print(f"Vertices: {n_vertex}, chromo: {list(ds['chromo'].values)}")
if "time" in ds.dims or "reltime" in ds.dims:
    print("WARNING: this file has a time/reltime dimension after all -- the "
          "assumption that image_recon.mag.enable=true (static, no time axis) "
          "doesn't hold for your config. This script doesn't handle that case; "
          "see plot_groupaverage_brainmaps.py's ANIMATE handling for the pattern "
          "to adapt.")


# %% ------------------------------------------------------------------------------
# LOAD HEAD MODEL (mesh geometry for plotting -- unrelated to the Adot sensitivity
# data loaded below) + BUILD SENSITIVITY MASK
# -----------------------------------------------------------------------------

print(f"Loading standard head model: {HEAD_MODEL}")
head = cedalion.dot.get_standard_headmodel(HEAD_MODEL)

print(f"Loading sensitivity matrix (Adot) for masking:\n  {ADOT_FILE}")
Adot = io.forward_model.load_Adot(ADOT_FILE)

if Adot.sizes["vertex"] != n_vertex:
    raise RuntimeError(
        f"Adot has {Adot.sizes['vertex']} vertices but the group-average file has "
        f"{n_vertex}. This data is already vertex-space (unlike the reconfirst "
        "pipeline's parcel-space output), so it must have been reconstructed "
        "against the SAME Adot as the one loaded here -- check ADOT_FILE points "
        "at the same forward model used by your image_recon rule."
    )

# Same convention as image_recon.py's own (already-working) per-run plots: a
# vertex only counts as "seen" if summed sensitivity across all channels (first
# wavelength) is above a floor. No brain-only restriction needed here (unlike the
# reconfirst script) -- this data already spans the same combined brain+scalp
# vertex set as Adot itself.
intensity = np.log10(Adot[:, :, 0].sum("channel"))
sensitivity_mask = (intensity > SENSITIVITY_LOG10_THRESH).drop_vars(
    "wavelength", errors="ignore"
)
n_sensed = int(sensitivity_mask.sum())
print(f"Sensitivity mask: {n_sensed}/{n_vertex} vertices pass the "
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
        data_tt = data_tt.where(sensitivity_mask)

        # keep 'chromo' as a real dim -- image_recon() (called inside
        # image_recon_multi_view) does its own X.sel(chromo='HbO'/'HbR')
        # internally regardless of view_type.
        data_tt = data_tt.transpose("vertex", ...)

        for chromo in CHROMO_LIST:
            clim_max = float(
                np.nanmax(np.abs(
                    data_tt.sel(chromo=chromo).pint.dequantify().values
                ))
            )
            clim = (-clim_max, clim_max)

            for view in VIEW_LIST:
                hbx = f"{'hbo' if chromo == 'HbO' else 'hbr'}_{view}"
                title_str = f"{trial_type} {chromo} {var_key} {view}"
                filename = f"GROUPAVG_{trial_type}_{var_key}_{hbx}"
                save_file_path = os.path.join(SAVE_DIR, filename)

                print(f"Rendering: {filename}")
                image_recon_multi_view(
                    data_tt,
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
