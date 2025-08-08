# -*- coding: utf-8 -*-

import xarray as xr
import cedalion
import numpy as np

def weighted_groupaverage(blockaverage_subj):
        
    # get the unweighted average
    blockaverage_mean = blockaverage_subj.mean('subj')
    
    # get the mean mse within subjects
    mse_mean_within_subject = 1 / sum_mse_inv
    
    blockaverage_mean_weighted = blockaverage_mean_weighted / sum_mse_inv
    
    mse_weighted_between_subjects_tmp = (blockaverage_subj - blockaverage_mean_weighted)**2 / mse_subj
    mse_weighted_between_subjects = mse_weighted_between_subjects_tmp.mean('subj')
    mse_weighted_between_subjects = mse_weighted_between_subjects * mse_mean_within_subject # normalized by the within subject variances as weights
    
    # get the weighted average
    mse_btw_within_sum_subj = mse_subj + mse_weighted_between_subjects
    denom = (1/mse_btw_within_sum_subj).sum('subj')
    
    blockaverage_mean_weighted = (blockaverage_subj / mse_btw_within_sum_subj).sum('subj')
    blockaverage_mean_weighted = blockaverage_mean_weighted / denom
    
    mse_total = 1/denom
    
    total_stderr_blockaverage = np.sqrt( mse_total )
    total_stderr_blockaverage = total_stderr_blockaverage.assign_coords(trial_type=blockaverage_mean_weighted.trial_type)
