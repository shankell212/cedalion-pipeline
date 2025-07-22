#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 17:44:18 2025

@author: smkelley
"""
import cedalion
import os
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt 
import yaml

#%%

root_data_dir = "/projectnb/nphfnirs/s/datasets/BSMW_Laura_Miray_2025/BS/derivatives/processed_data/"
filepath = os.path.join(root_data_dir, "od_o_postglm_ind_blockaverage_for_IR.pkl.gz")

rec_str = "od_o_postglm"

print("Loading saved data")
with gzip.open( os.path.join(root_data_dir, rec_str+'_ind_blockaverage_for_IR.pkl.gz'), 'rb') as f:
     all_results = pickle.load(f)
     
blockaverage = all_results['blockaverage']  # group_blockaverage unweighted rename
blockaverage_weighted = all_results['blockaverage_weighted']
blockaverage_stderr = all_results['blockaverage_stderr']
blockaverage_subj = all_results['blockaverage_subj'] # always unweighted   - load into img recon
blockaverage_mse_subj = all_results['blockaverage_mse_subj'] # - load into img recon
geo2d = all_results['geo2d']
geo3d = all_results['geo3d']



# Plot mse_hist
mse_subj_stacked = blockaverage_mse_subj.stack(foo=['subj', 'trial_type', 'channel', 'wavelength', 'reltime'])

f, ax = plt.subplots()
ax.hist(np.log10(mse_subj_stacked), bins=100)


source = 'S55'
matching_channels = [ch for ch in blockaverage_weighted.channel.values if source in ch]

TRIAL = 'right'
CHANNEL = matching_channels[8] #'S5D137'
WAV = 850  #760
grp_right = blockaverage_weighted.sel(wavelength=WAV, channel=CHANNEL, trial_type='right')
grp_left = blockaverage_weighted.sel(wavelength=WAV, channel=CHANNEL, trial_type='left')

f, ax = plt.subplots(1,1)
ax.plot(grp_right.reltime, grp_right, label='right')
ax.plot(grp_left.reltime, grp_left, label='left')

plt.ylabel('OD')
plt.xlabel('time (s)')
plt.legend()
plt.title(f'group avg weighted (wav {WAV}, channel {CHANNEL}')
        





#%% Old block avg code


# # Loop over subjects
# #pdb.set_trace()
# blockaverage_subj = None
# for subj_idx, subj in enumerate(blockavg_files):

#     # Load in json data qual
#     with open(data_quality_files[subj_idx], 'r') as fp:
#         data_quality_sub = json.load(fp)
#     idx_amp = np.array(data_quality_sub['idx_amp'])
#     idx_sat = np.array(data_quality_sub['idx_sat'])
#     bad_chans_sat = np.array(data_quality_sub['bad_chans_sat'])
#     bad_chans_amp = np.array(data_quality_sub['bad_chans_amp'])
    
#     # Load in current sub's blockaverage file
#     with open(subj, 'rb') as f:
#         rec_blockavg = pickle.load(f)
#     blockaverage = rec_blockavg['blockaverage']
#     epochs_all = rec_blockavg['epochs']
    
#     # Load geometric positions and landmarks # !!! don't need to do for each subject in reality, but for snakemake yes?
#     with gzip.open(geo_files[subj_idx], 'rb') as f:
#         geo_pos = pickle.load(f)
#     geo2d = geo_pos['geo2d']
#     geo3d = geo_pos['geo3d']
  
#     # Load in net cdf files
#     #blockaverage2 = xr.load_dataarray(blockavg_files_nc)
#     #epochs2 = xr.load_dataarray(epoch_files_nc)
    
#     blockaverage_weighted = blockaverage.copy()
    
#     n_epochs = len(epochs_all.epoch)
#     n_chs = len(epochs_all.channel)

#     mse_t_lst = []
    
# #Loop thru trial tpes

#     # Loop thru trial tpes
  
#     for idxt, trial_type in enumerate(cfg_hrf['stim_lst']): 
        
#         epochs_zeromean = epochs_all.where(epochs_all.trial_type == trial_type, drop=True) - blockaverage_weighted.sel(trial_type=trial_type) # zero mean data

#         if 'chromo' in blockaverage.dims: # !!! save 2d and 3d pts in blockaverage????

#             foo_t = epochs_zeromean.stack(measurement=['channel','chromo']).sortby('chromo')
#         else:
#             foo_t = epochs_zeromean.stack(measurement=['channel','wavelength']).sortby('wavelength')

#         foo_t = foo_t.transpose('measurement', 'reltime', 'epoch')  # !!! this does not have trial type?
#         mse_t = (foo_t**2).sum('epoch') / (n_epochs - 1)**2 # this is squared to get variance of the mean, aka MSE of the mean
        
#         # ^ this gets the variance  across epochs
        

#         # # get the variance, correctig channels we don't trust    
#         # mse_t = quality.measurement_variance( # this gets the measurement variance across time ... only good for timeseries
#         #         foo_t,
#         #         list_bad_channels = None, #idx_bad_channels,  # !!! where to get this?  -- before this was where mse_t = 0
#         #         bad_rel_var = 1e6, # If bad_abs_var is none then it uses this value relative to maximum variance
#         #         bad_abs_var = None, #cfg_blockavg['cfg_mse_conc']['mse_val_for_bad_data'],
#         #         calc_covariance = False
#         #     )
        
#         # mse_t = blockaverage_weighted + cfg_mse['mse_min_thresh'] # set a small value to avoid dominance for low variance channels
#         # blockaverage_weighted.loc[dict(channel=blockaverage_weighted.isel(channel=idx_amp).channel.data)] = cfg_mse['blockaverage_val']
#         # blockaverage_weighted.loc[dict(channel=blockaverage_weighted.isel(channel=idx_sat).channel.data)] = cfg_mse['blockaverage_val']

#         # !!! maybe have the above func have a helper func that is called variance_clean  

#         # Set bad values in mse_t to the bad value threshold
#         bad_mask = mse_t.data == 0
#         bad_any = bad_mask.any(axis=1)
#         bad_chans_mse = mse_t.channel[bad_any].values
#         # bad_chans_mse = set(mse_t.channel[bad_any].values)

#         # idx_bad = np.where(mse_t == 0)[0]
#         # idx_bad1 = idx_bad[idx_bad<n_chs]
#         # idx_bad2 = idx_bad[idx_bad>=n_chs] - n_chs
        
#         # mse_t[idx_amp,:] = cfg_mse['mse_val_for_bad_data']
#         # mse_t[idx_amp+n_chs,:] = cfg_mse['mse_val_for_bad_data']
#         # mse_t[idx_sat,:] = cfg_mse['mse_val_for_bad_data']    

#         # mse_t[idx_sat+n_chs,:] = cfg_mse['mse_val_for_bad_data']
     
#         # mse_t.loc[dict(channel=bad_chans_amp)] = cfg_mse['mse_val_for_bad_data']
#         # mse_t.loc[dict(channel=bad_chans_sat)] = cfg_mse['mse_val_for_bad_data']
#         mse_t[mse_t.channel.isin(bad_chans_amp), :] = cfg_mse['mse_val_for_bad_data']
#         mse_t[mse_t.channel.isin(bad_chans_sat), :] = cfg_mse['mse_val_for_bad_data']
#         mse_t[mse_t.channel.isin(bad_chans_mse), :] = cfg_mse['mse_val_for_bad_data']
#         # mse_t[idx_bad] = cfg_mse['mse_val_for_bad_data']
#         # mse_t.loc[dict(channel=bad_chans_mse)] = cfg_mse['mse_val_for_bad_data']
        
#         channels = blockaverage_weighted.channel
#         blockaverage_weighted.loc[dict(trial_type=trial_type, channel=bad_chans_sat)] = cfg_mse['blockaverage_val']
#         blockaverage_weighted.loc[dict(trial_type=trial_type, channel=bad_chans_amp)] = cfg_mse['blockaverage_val']
#         blockaverage_weighted.loc[dict(trial_type=trial_type, channel=bad_chans_mse)] = cfg_mse['blockaverage_val']
#         #blockaverage_weighted.loc[dict(trial_type=trial_type, channel=bad_chans_mse)] = cfg_mse['blockaverage_val']
        
#         blockaverage.loc[dict(trial_type=trial_type, channel=bad_chans_amp)] = cfg_mse['blockaverage_val']
#         blockaverage.loc[dict(trial_type=trial_type, channel=bad_chans_sat)] = cfg_mse['blockaverage_val']
#         blockaverage.loc[dict(trial_type=trial_type, channel=bad_chans_mse)] = cfg_mse['blockaverage_val']
#         #blockaverage.loc[dict(trial_type=trial_type, channel=bad_chans_mse)] = cfg_mse['blockaverage_val']


#         # !!! DO we still wanna do this?
#         # set the minimum value of mse_t
#         mse_t = xr.where(mse_t < cfg_mse['mse_min_thresh'], cfg_mse['mse_min_thresh'], mse_t)
        
#         if 'chromo' in blockaverage.dims:
#             mse_t = mse_t.unstack('measurement').transpose('chromo','channel','reltime')  
#         else:
#             mse_t = mse_t.unstack('measurement').transpose('wavelength','channel','reltime')  # !!! xrutils.other_dim
            
#         mse_t = mse_t.expand_dims('trial_type')
#         source_coord = blockaverage_weighted['source']
#         mse_t = mse_t.assign_coords(source=('channel',source_coord.data))
#         detector_coord = blockaverage_weighted['detector']
#         mse_t = mse_t.assign_coords(detector=('channel',detector_coord.data))
        
        
#         mse_t = mse_t.assign_coords(trial_type = [trial_type]) # assign coords to match curr trial type
#         mse_t_lst.append(mse_t) # append mse_t for curr trial type to list

#         # DONE LOOP OVER TRIAL TYPES
        
#     mse_t_tmp = xr.concat(mse_t_lst, dim='trial_type') # concat the 2 trial types
#     mse_t = mse_t_tmp # reassign the newly appended mse_t with both trial types to mse_t 

    
#     # gather the blockaverage across subjects
#     if blockaverage_subj is None: 
#         # add a subject dimension and coordinate
#         blockaverage_subj = blockaverage.expand_dims('subj')
#         blockaverage_subj = blockaverage_subj.assign_coords(subj=[cfg_dataset['subject'][subj_idx]]) # !!! will need to update when exluding subs

#         blockaverage_mse_subj = mse_t.expand_dims('subj') # mse of blockaverage for each sub
#         blockaverage_mse_subj = blockaverage_mse_subj.assign_coords(subj=[cfg_dataset['subject'][subj_idx]]) # !!! does snakemake give list of files in order of sub list?
        
#         blockaverage_mean_weighted = blockaverage_weighted / mse_t

#         blockaverage_mse_inv_mean_weighted = 1 / mse_t
        
#     else:   
#         #blockaverage_subj_tmp = blockaverage_weighted
#         blockaverage_subj_tmp = blockaverage.expand_dims('subj')
#         blockaverage_subj_tmp = blockaverage_subj_tmp.assign_coords(subj=[cfg_dataset['subject'][subj_idx]])
#         blockaverage_subj = xr.concat([blockaverage_subj, blockaverage_subj_tmp], dim='subj')

#         blockaverage_mse_subj_tmp = mse_t.expand_dims('subj')
        
#         blockaverage_mse_subj_tmp = blockaverage_mse_subj_tmp.assign_coords(subj=[cfg_dataset['subject'][subj_idx]])
#         blockaverage_mse_subj = xr.concat([blockaverage_mse_subj, blockaverage_mse_subj_tmp], dim='subj') # !!! this does not have trial types

#         blockaverage_mean_weighted = blockaverage_mean_weighted +  blockaverage_weighted / mse_t
#         blockaverage_mse_inv_mean_weighted = blockaverage_mse_inv_mean_weighted + 1/mse_t 
    
#     # DONE LOOP OVER SUBJECTS
    
# # get the unweighted average
# blockaverage_mean = blockaverage_subj.mean('subj')

# # get the weighted average  (old)
# #blockaverage_mean_weighted = blockaverage_mean_weighted / blockaverage_mse_inv_mean_weighted

# # get the mean mse within subjects
# mse_mean_within_subject = 1 / blockaverage_mse_inv_mean_weighted

# # get the mse between subjects
# mse_weighted_between_subjects_tmp = (blockaverage_subj - blockaverage_mean_weighted)**2 / blockaverage_mse_subj   # was -blockaverage_mean_weighted
# mse_weighted_between_subjects = mse_weighted_between_subjects_tmp.mean('subj')
# mse_weighted_between_subjects = mse_weighted_between_subjects * mse_mean_within_subject  # normalized by the within subject variances as weights

# # get the weighted average
# mse_btw_within_sum_subj = blockaverage_mse_subj + mse_weighted_between_subjects
# denom = (1/mse_btw_within_sum_subj).sum('subj')

# blockaverage_mean_weighted = (blockaverage_subj / mse_btw_within_sum_subj).sum('subj') # !!! reassinging blockaverage_mean_weighted
# blockaverage_mean_weighted = blockaverage_mean_weighted / denom

# mse_total = 1/denom

# total_stderr_blockaverage = np.sqrt( mse_total )
# total_stderr_blockaverage = total_stderr_blockaverage.assign_coords(trial_type=blockaverage_mean_weighted.trial_type)



# # # !!! Do we want these plots still? Would need to also load in a rec ???  - or just load in saved geo2d and geo3d?
# # # # Plot scalp plot of mean, tstat,rsme + Plot mse hist
# # # for idxt, trial_type in enumerate(blockaverage_mean_weighted.trial_type.values):         
# # #     plot_mean_stderr(rec, rec_str, trial_type, cfg_dataset, cfg_blockavg, blockaverage_mean_weighted, 
# # #                      total_stderr_blockaverage, mse_mean_within_subject, mse_weighted_between_subjects)
# # #     plot_mse_hist(rec, rec_str, trial_type, cfg_dataset, blockaverage_mse_subj, cfg_blockavg['mse_val_for_bad_data'], cfg_blockavg['mse_min_thresh'])  # !!! not sure if these r working correctly tbh


# groupavg_results = {'group_blockaverage_weighted': blockaverage_mean_weighted, # weighted group avg   
#             'group_blockaverage': blockaverage_mean,  # unweighted group aaverage
#            'total_stderr_blockaverage': total_stderr_blockaverage,
#            'blockaverage_subj': blockaverage_subj,  # always unweighted   - load into img recon
#            'blockaverage_mse_subj': blockaverage_mse_subj, # - load into img recon
#            'geo2d' : geo2d,
#            'geo3d' : geo3d     # !!! save 2d and 3d pts in blockaverage????
#            }

# # Save data a pickle for now  # !!! Change to snirf in future when its debugged
# with open(out, "wb") as f:        # if output is a single string, it wraps it in an output object and need to index in
#     pickle.dump(groupavg_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
# print("Group average data saved successfully")