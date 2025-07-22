#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 13:55:46 2025

@author: smkelley
"""

import os
import gzip
import pickle
import tkinter as tk
from tkinter import filedialog

import cedalion.vis.time_series as vTS

#%%

root_dir = "/projectnb/nphfnirs/s/datasets/Interactive_Walking_HD"
deriv_dir = 'cedalion'  # CHANGE
task = 'STS'  # CHANGE

preproc_dir = os.path.join(root_dir, 'derivatives', deriv_dir, 'preprocessed_data')

rec_pkl = os.path.join(preproc_dir, f'rec_list_ts_{task}.pkl')
chs_pruned_subjs_pkl = os.path.join(preproc_dir, f'chs_pruned_subjs_ts_{task}.pkl')

#%%
with gzip.open(rec_pkl, 'rb') as f:
    rec = pickle.load(f)

#%%
with gzip.open(chs_pruned_subjs_pkl, 'rb') as f:
    chs_pruned_subjs = pickle.load(f)
    
#%%
vTS.run_vis(rec[0][0])


#%%
# # does .pkl.gz exist in the current directory?
# if os.path.exists('rec.pkl.gz'):
#     with gzip.open('rec.pkl.gz', 'rb') as f:
#         rec  = pickle.load(f)
#     vTS.run_vis(rec)

# else:

#     # set initialdir to current directory
#     initialdir = os.getcwd()

#     # Create a Tkinter root window (it will not be shown)
#     root = tk.Tk()
#     root.withdraw()

#     # Open a file dialog to select a file
#     file_path = filedialog.askopenfilename(
#         initialdir=initialdir,
#         title='Select a data file',
#         filetypes=(('gZip files', '*.gz'), ('Pickle files', '*.pkl'), ('All files', '*.*'))
#     )

#     # Check if a file was selected
#     if file_path:
#         with gzip.open(file_path, 'rb') as f:
#             rec = pickle.load(f)

#         vTS.run_vis(rec)
#     else:
#         print("No file selected.")