import cedalion
import cedalion.datasets as datasets
import cedalion.imagereco.forward_model as fw
import cedalion.io as io
import cedalion.nirs as nirs
import xarray as xr
from cedalion import units
import cedalion.dataclasses as cdc 
import numpy as np
import os.path
import pickle
from cedalion.imagereco.solver import pseudo_inverse_stacked
import cedalion.xrutils as xrutils

import matplotlib.pyplot as p
import pyvista as pv
from matplotlib.colors import ListedColormap

import gzip

import sys

import module_spatial_basis_funs as sbf 
import pdb

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




#%% MATRIX CALCULATIONS

def get_Adot_scaled(Adot, wavelengths, BRAIN_ONLY=False):
    
    if BRAIN_ONLY:
        Adot = Adot[:, Adot.is_brain.values, :] 

    nchannel = Adot.shape[0]
    nvertices = Adot.shape[1]
    E = nirs.get_extinction_coefficients('prahl', Adot.wavelength)

    A = np.zeros((2 * nchannel, 2 * nvertices))
    wl1 = wavelengths[0]
    wl2 = wavelengths[1]
    A[:nchannel, :nvertices] = E.sel(chromo="HbO", wavelength=wl1).values * Adot.sel(wavelength=wl1) # noqa: E501
    A[:nchannel, nvertices:] = E.sel(chromo="HbR", wavelength=wl1).values * Adot.sel(wavelength=wl1) # noqa: E501
    A[nchannel:, :nvertices] = E.sel(chromo="HbO", wavelength=wl2).values * Adot.sel(wavelength=wl2) # noqa: E501
    A[nchannel:, nvertices:] = E.sel(chromo="HbR", wavelength=wl2).values * Adot.sel(wavelength=wl2) # noqa: E501

    A = xr.DataArray(A, dims=("measurement", "flat_vertex"))
    A = A.assign_coords({"parcel" : ("flat_vertex", np.concatenate((Adot.coords['parcel'].values, Adot.coords['parcel'].values))),
                         "is_brain" : ("flat_vertex", np.concatenate((Adot.coords['is_brain'].values, Adot.coords['is_brain'].values)))})
    
    return A

def calculate_W(A, alpha_meas=0.1, alpha_spatial=0.01, BRAIN_ONLY = False, DIRECT=True, C_meas_flag=False, C_meas=None, D=None, F=None):
    
    
    if DIRECT:
        if C_meas_flag:
            C_meas = np.diag(C_meas)

        W_xr, D, F = _calculate_W_direct(A, alpha_meas=alpha_meas, alpha_spatial=alpha_spatial, 
                                        BRAIN_ONLY=BRAIN_ONLY, 
                                        C_meas_flag=C_meas_flag, C_meas=C_meas, D=D, F=F)
    else:
        W_xr, D, F = _calculate_W_indirect(A, alpha_meas=alpha_meas, alpha_spatial=alpha_spatial,
                                          BRAIN_ONLY=BRAIN_ONLY, 
                                          C_meas_flag=C_meas_flag, C_meas=C_meas, D=D, F=F)
        
    return W_xr, D, F

def _calculate_W_direct(A, alpha_meas=0.1, alpha_spatial=0.01, BRAIN_ONLY=False, 
                       C_meas_flag=False, C_meas=None, D=None, F=None):
    
    A_coords = A.coords
    A = A.pint.dequantify().values
                
    if BRAIN_ONLY:
    
        B = A@A.T
        b = max(np.diag(B))
        W = A.T @ np.linalg.pinv(B + (alpha_meas * b.values * np.eye(np.shape(B)[0])))
    
    else:
        if D is None and F is None:
            B = np.sum((A ** 2), axis=0)
            b = B.max()
            
            # GET A_HAT
            lambda_spatial = alpha_spatial * b
            
            L = np.sqrt(B + lambda_spatial)
            Linv = 1/L
            
            A_hat = A * Linv
            
            #% GET W
            F = A_hat @ A_hat.T
    
            D = Linv[:, np.newaxis]**2 * A.T
        else:
            D = D.values
            F = F.values
            
        max_eig = np.max(np.linalg.eigvals(F))
        lambda_meas = alpha_meas * max_eig
        
        if C_meas_flag:  
            #pdb.set_trace()
            assert  len(C_meas.shape) == 2
                                                            
            W = D @ np.linalg.inv(F  + lambda_meas * C_meas )
        else:
            W = D @ np.linalg.inv(F  + lambda_meas * np.eye(A.shape[0]) )
        
        W_xr = xr.DataArray(W, dims=("flat_vertex", "measurement"))
        D_xr = xr.DataArray(D, dims=("flat_vertex", "measurement"))

        if 'parcel' in A_coords:
            W_xr = W_xr.assign_coords({"parcel" : ("flat_vertex", A_coords['parcel'].values)})
            D_xr = D_xr.assign_coords({"parcel" : ("flat_vertex", A_coords['parcel'].values)})
        if 'is_brain' in A_coords:
            W_xr = W_xr.assign_coords({"is_brain": ("flat_vertex", A_coords['is_brain'].values)}) 
            D_xr = D_xr.assign_coords({"is_brain": ("flat_vertex", A_coords['is_brain'].values)})

        
        F_xr = xr.DataArray(F, dims=("measurement1", "measurement2"))

    return W_xr, D_xr, F_xr

def _calculate_W_indirect(A, alpha_meas=0.1, alpha_spatial=0.01, BRAIN_ONLY=False, 
                       C_meas_flag=False, C_meas=None, D=None, F=None):
    
    
    W = []
    D_lst = []
    F_lst = []
    for wavelength in A.wavelength:
        
        if C_meas_flag:
            C_meas_wl = C_meas.sel(wavelength=wavelength)
            C_meas_wl = np.diag(C_meas_wl)
        else:
            C_meas_wl = None
            
        A_wl = A.sel(wavelength=wavelength)
        if D is None and F is None:
            W_wl, D_wl, F_wl = _calculate_W_direct(A_wl, alpha_meas=alpha_meas, alpha_spatial=alpha_spatial,BRAIN_ONLY=BRAIN_ONLY,
                                      C_meas_flag=C_meas_flag, C_meas=C_meas_wl, D=D, F=F)
        else:
            W_wl, D_wl, F_wl = _calculate_W_direct(A_wl, alpha_meas=alpha_meas, alpha_spatial=alpha_spatial,BRAIN_ONLY=BRAIN_ONLY,
                                      C_meas_flag=C_meas_flag, C_meas=C_meas_wl, 
                                      D=D.sel(wavelength=wavelength), F=F.sel(wavelength=wavelength))
        
        
        W.append(W_wl)
        D_lst.append(D_wl)
        F_lst.append(F_wl)
    
    W_xr = xr.concat(W, dim='wavelength')
    W_xr = W_xr.assign_coords(wavelength=A.wavelength)
    
    D = xr.concat(D_lst, dim='wavelength')
    D = D.assign_coords(wavelength=A.wavelength)

    F = xr.concat(F_lst, dim='wavelength')
    F = F.assign_coords(wavelength=A.wavelength)

    return W_xr, D, F

#%% do image recon
def _get_image_brain_scalp_direct(y, W, A, SB=False, G=None):
    
    y = y.stack(measurement=['channel', 'wavelength']).sortby('wavelength')
    try:
        X = W.values @ y.values
    except:
        X = W @ y
        X = X.values
        
    split = len(X)//2
    
    if SB:
        X = sbf.go_from_kernel_space_to_image_space_direct(X, G)

    
    else:
        if len(X.shape) == 1:
            X = X.reshape([2, split]).T
        else:
            X = X.reshape([2, split, X.shape[1]])
            X = X.transpose(1,2,0)
    
    if len(X.shape) == 2:
        X = xr.DataArray(X, 
                         dims = ('vertex', 'chromo'),
                         coords = {'chromo': ['HbO', 'HbR']}
                         )
    else:
        if 'time' in y.dims:
            t = y.time
            t_name = 'time'
        elif 'reltime' in y.dims:
            t = y.reltime
            t_name = 'reltime'
        X = xr.DataArray(X, 
                         dims = ('vertex',  t_name, 'chromo',),
                         coords = {'chromo': ['HbO', 'HbR'],
                                   t_name: t},
                         )
    if 'parcel' in A.coords:
        X = X.assign_coords({"parcel" : ("vertex", A.coords['parcel'].values)})
                              
    if 'is_brain' in A.coords:
        X = X.assign_coords({"is_brain": ("vertex", A.coords['is_brain'].values)}) 

    return X


def _get_image_brain_scalp_indirect(y, W, A, SB=False, G=None):
                  
     W_indirect_wl0 = W.isel(wavelength=0)
     W_indirect_wl1 = W.isel(wavelength=1)
     
     y_wl0 = y.isel(wavelength=0) #[:split]
     y_wl1 = y.isel(wavelength=1) #[split:]

     try:
        X_wl0 = W_indirect_wl0.values @ y_wl0.values
        X_wl1 = W_indirect_wl1.values @ y_wl1.values
     except:
        X_wl0 = W_indirect_wl0 @ y_wl0
        X_wl1 = W_indirect_wl1 @ y_wl1   
        
             
     if SB:
         X_wl0 = sbf.go_from_kernel_space_to_image_space_indirect(X_wl0, G)
         X_wl1 = sbf.go_from_kernel_space_to_image_space_indirect(X_wl1, G)
         
      
     X_od = np.stack([X_wl0, X_wl1], axis=1)  
    
     if len(X_od.shape) == 2:
         X_od = xr.DataArray(X_od, 
                          dims = ('vertex', 'wavelength'),
                          coords = {'wavelength': W.wavelength}
                          )
     else:
         if 'time' in y.dims:
             t = y.time
             t_name = 'time'
         elif 'reltime' in y.dims:
             t = y.reltime
             t_name = 'reltime'
         X_od = xr.DataArray(X_od, 
                          dims = ('vertex', 'wavelength', t_name),
                          coords = {'wavelength': W.wavelength,
                                    t_name: t},
                          )

     if 'parcel' in A.coords:
        X_od = X_od.assign_coords({"parcel" : ("vertex", A.coords['parcel'].values)})
                              
     if 'is_brain' in A.coords:
        X_od = X_od.assign_coords({"is_brain": ("vertex", A.coords['is_brain'].values)}) 


     # convert to concentration 
     E = nirs.get_extinction_coefficients('prahl', W.wavelength)
     einv = xrutils.pinv(E) 

     X = xr.dot(einv, X_od/units.mm, dims=["wavelength"])
     
     return X

def do_image_recon(od, head, Adot, C_meas_flag, C_meas, wavelength, BRAIN_ONLY, DIRECT,
                   SB, cfg_sbf, alpha_spatial, alpha_meas, D, F, G ):
    
    
    if DIRECT:
        Adot_stacked = get_Adot_scaled(Adot, wavelength)
        
        if SB:
            if G is None:
                M = sbf.get_sensitivity_mask(Adot, cfg_sbf['mask_threshold'], 1)
                G = sbf.get_G_matrix(head, M, threshold_brain=cfg_sbf['threshold_brain'],
                                             threshold_scalp = cfg_sbf['threshold_scalp'],
                                             sigma_brain=cfg_sbf['sigma_brain'],
                                             sigma_scalp=cfg_sbf['sigma_scalp'])
                
            H_stacked = sbf.get_H_stacked(G, Adot_stacked)
            Adot_stacked = H_stacked.copy()
            
        W, D, F = calculate_W(Adot_stacked, alpha_meas=alpha_meas, alpha_spatial=alpha_spatial,
                              C_meas_flag=C_meas_flag, C_meas=C_meas, DIRECT=DIRECT, BRAIN_ONLY=BRAIN_ONLY, D=D, F=F)
       
        X = _get_image_brain_scalp_direct(od, W, Adot, SB=SB, G=G)
    
            
    else:
        if SB:
            if G is None:
                M = sbf.get_sensitivity_mask(Adot, cfg_sbf['mask_threshold'], 1)
                G = sbf.get_G_matrix(head, M, threshold_brain=cfg_sbf['threshold_brain'],
                                             threshold_scalp = cfg_sbf['threshold_scalp'],
                                             sigma_brain=cfg_sbf['sigma_brain'],
                                             sigma_scalp=cfg_sbf['sigma_scalp'])
            
            H = sbf.get_H(G, Adot)
            Adot_ir = H.copy()
            
        else:
            Adot_ir = Adot.copy()

            
        W, D, F = calculate_W(Adot_ir, alpha_meas=alpha_meas, alpha_spatial=alpha_spatial,
                              C_meas_flag=C_meas_flag, C_meas=C_meas, DIRECT=DIRECT, BRAIN_ONLY=BRAIN_ONLY, D=D, F=F)
        X = _get_image_brain_scalp_indirect(od, W, Adot, SB=SB, G=G)

    if len(od.shape) == 3:
        if 'time' in X.dims:
            X = X.transpose('chromo', 'vertex', 'time')
        elif 'reltime' in X.dims:
            X = X.transpose('chromo', 'vertex', 'reltime')
    else:
        X = X.transpose('chromo', 'vertex')

    return X, W, D, F, G


def get_image_noise(C_meas, X, W, SB=False, DIRECT=True, G=None):

    TIME = False
    if 'time' in C_meas.dims:
        t_dim = 'time'
        TIME = True
    elif 'reltime' in C_meas.dims:
        t_dim = 'reltime'
        TIME = True

    if DIRECT:
        if TIME:
            C_tmp_lst = []
            for t in C_meas[t_dim]:
                    C_tmp = C_meas.sel({t_dim:t})
                    cov_img_tmp = W * np.sqrt(C_tmp.values) # get diag of image covariance
                    cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
                    C_tmp_lst.append(cov_img_diag)

            cov_img_diag = np.vstack(C_tmp_lst)
        else:
            cov_img_tmp = W *np.sqrt(C_meas.values) # W is pseudo inverse  --- diagonal (faster than W C W.T)
            cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
        
        if SB:
            cov_img_diag = sbf.go_from_kernel_space_to_image_space_direct(cov_img_diag.T, G)
        else:
            if TIME:
                split = cov_img_diag.shape[1]//2
                HbO = cov_img_diag[:, :split]   # shape: (time, vertices)
                HbR = cov_img_diag[:, split:]   # shape: (time, vertices)

                # Stack into vertex x 2 x time
                cov_img_diag = np.stack([HbO.T, HbR.T], axis=1)  # (vertex, 2, time)
                cov_img_diag = cov_img_diag.transpose(1,0,2)
            else:
                split = len(cov_img_diag)//2
                cov_img_diag =  np.reshape( cov_img_diag, (2,split) )

        

    else:
        cov_img_lst = []
        E = nirs.get_extinction_coefficients('prahl', W.wavelength)
        einv = xrutils.pinv(E)

        for wavelength in W.wavelength:
            W_wl = W.sel(wavelength=wavelength)
            C_wl = C_meas.sel(wavelength=wavelength)

            if TIME:
                C_tmp_lst = []
                for t in C_wl[t_dim]:
                    C_tmp = C_wl.sel({t_dim:t})
                    cov_img_tmp = W_wl * np.sqrt(C_tmp.values) # get diag of image covariance
                    cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
                    C_tmp_lst.append(cov_img_diag)

                cov_img_diag = np.vstack(C_tmp_lst).T

            else:
                cov_img_tmp = W_wl * np.sqrt(C_wl.values) # get diag of image covariance
                cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
            
            if SB:
                cov_img_diag = sbf.go_from_kernel_space_to_image_space_indirect(cov_img_diag, G)
                # cov_img_diag = cov_img_diag

            cov_img_lst.append(cov_img_diag)
            
        if TIME:
            cov_img_diag =  np.stack(cov_img_lst, axis=2)
            cov_img_diag = np.transpose(cov_img_diag, [2,0,1])
            cov_img_diag = np.einsum('ij,jab->iab', einv.values**2, cov_img_diag)

        else:
            cov_img_diag =  np.vstack(cov_img_lst) 
            cov_img_diag = einv.values**2 @ cov_img_diag
            
    noise = X.copy()
    noise.values = cov_img_diag

    return noise

def get_Adot_parcels( Adot = None ):

    # reduce parcels to 17 network parcels plus 'Background+Freesurfer...'
    # get the unique 17 network parcels and remove non-brain parcels
    unique_parcels = Adot.groupby('parcel').sum('vertex').parcel
    unique_parcels = unique_parcels.sel(parcel=unique_parcels.parcel != 'scalp')
    unique_parcels = unique_parcels.sel(parcel=unique_parcels.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_LH')
    unique_parcels = unique_parcels.sel(parcel=unique_parcels.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_RH')


    parcel_list = []
    for parcel in unique_parcels.values:
        parcel_list.append( parcel.split('_')[0] + '_' + parcel.split('_')[-1] )
    unique_parcels_lev1 = np.unique(parcel_list)

    parcel_list_lev2 = []
    for parcel in unique_parcels.values:
        if parcel.split('_')[1].isdigit():
            parcel_list_lev2.append( parcel.split('_')[0] + '_' + parcel.split('_')[-1] )
        else:
            parcel_list_lev2.append( parcel.split('_')[0] + '_' + parcel.split('_')[1] + '_' + parcel.split('_')[-1] )
    unique_parcels_lev2 = np.unique(parcel_list_lev2)


    Adot_parcels = Adot.isel(wavelength=0).groupby('parcel').sum('vertex')
    Adot_parcels = Adot_parcels.sel(parcel=Adot_parcels.parcel != 'scalp')
    Adot_parcels = Adot_parcels.sel(parcel=Adot_parcels.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_LH')
    Adot_parcels = Adot_parcels.sel(parcel=Adot_parcels.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_RH')


    Adot_parcels_lev1 = np.zeros( (Adot.shape[0], len(unique_parcels_lev1)) )
    for ii in range( 0, len(unique_parcels_lev1) ):
        idx1 = [i for i, x in enumerate(parcel_list) if x == unique_parcels_lev1.tolist()[ii]]
        Adot_parcels_lev1[:,ii] = np.sum(Adot_parcels.isel(parcel=idx1).values, axis=1)

    Adot_parcels_lev1_xr = xr.DataArray(
        Adot_parcels_lev1,
        dims=['channel','parcel'],
        coords={'channel': Adot.channel, 'parcel': unique_parcels_lev1}
    )


    Adot_parcels_lev2 = np.zeros( (Adot.shape[0], len(unique_parcels_lev2)) )
    for ii in range( 0, len(unique_parcels_lev2) ):
        idx1 = [i for i, x in enumerate(parcel_list_lev2) if x == unique_parcels_lev2.tolist()[ii]]
        Adot_parcels_lev2[:,ii] = np.sum(Adot_parcels.isel(parcel=idx1).values,axis=1)

    Adot_parcels_lev2_xr = xr.DataArray(
        Adot_parcels_lev2,
        dims=['channel','parcel'],
        coords={'channel': Adot.channel, 'parcel': unique_parcels_lev2}
    )

    return Adot_parcels_lev1_xr, Adot_parcels_lev2_xr, unique_parcels_lev1, unique_parcels_lev2

