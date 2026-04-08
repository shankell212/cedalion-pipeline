import cedalion
import cedalion.data as datasets
import cedalion.dot.forward_model as fw
import cedalion.io as io
import cedalion.nirs as nirs
import xarray as xr
from cedalion import units
# import cedalion.dataclasses as cdc 
import numpy as np
import os.path
# from cedalion.imagereco.solver import pseudo_inverse_stacked
import pickle
import cedalion.xrutils as xrutils
import module_spatial_basis_funs as sbf 

#%% DATA LOADING

def load_head_model(head_model='ICBM152', with_parcels=True):
    
    if head_model == 'ICBM152':
        SEG_DATADIR, mask_files, landmarks_file = datasets.get_icbm152_segmentation()
        if with_parcels:
            PARCEL_DIR = datasets.get_icbm152_parcel_file()
        else :
            PARCEL_DIR = None
            
    elif head_model == 'colin27':
        SEG_DATADIR, mask_files, landmarks_file = datasets.get_colin27_segmentation()()
        if with_parcels:
            PARCEL_DIR = datasets.get_colin27_parcel_file()
        else :
            PARCEL_DIR = None
            
    masks, t_ijk2ras = io.read_segmentation_masks(SEG_DATADIR, mask_files)

    
    head = fw.TwoSurfaceHeadModel.from_surfaces(
        segmentation_dir=SEG_DATADIR,
        mask_files = mask_files,
        brain_surface_file= os.path.join(SEG_DATADIR, "mask_brain.obj"),
        scalp_surface_file= os.path.join(SEG_DATADIR, "mask_scalp.obj"),
        landmarks_ras_file=landmarks_file,
        smoothing=0,
        fill_holes=True,
        parcel_file=PARCEL_DIR
    ) 
    head.scalp.units = units.mm
    # head.scalp.vertices = head.scalp.vertices * units.mm
    head.brain.units = units.mm
    # head.brain.vertices = head.brain.vertices * units.mm
    
    return head, PARCEL_DIR


def load_probe(probe_path, snirf_name ='fullhead_56x144_System2.snirf', head_model='ICBM152'):
    # pdb.set_trace()
    # with open(os.path.join(probe_path, 'fw',  head_model, 'Adot.pkl'), 'rb') as f:
    #     Adot = pickle.load(f)
    
    Adot = io.forward_model.load_Adot(os.path.join(probe_path, 'fw', head_model, 'Adot.nc'))
        
    recordings = io.read_snirf(probe_path + snirf_name)
    rec = recordings[0]
    geo3d = rec.geo3d
    amp = rec['amp']
    meas_list = rec._measurement_lists['amp']

    return Adot, meas_list, geo3d, amp


