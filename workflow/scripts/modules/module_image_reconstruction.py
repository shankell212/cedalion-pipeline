# Functions for image reconstruction

#%% Imports
import cedalion
import cedalion.dot as dot
import cedalion.xrutils as xrutils
from cedalion import units
from cedalion.io.forward_model import load_Adot
from cedalion.sigproc.quality import measurement_variance
from cedalion.vis.anatomy import image_recon_multi_view
import xarray as xr
import numpy as np


#%% G matrix for spatial basis

def get_G_matrix(head_ras, Adot):
    """Get the G matrix which contains all the information of the spatial basis """
    sbf = dot.GaussianSpatialBasisFunctions(
        head_ras,
        Adot,
        mask_threshold=-2,
        threshold_brain=1 * units.mm,
        threshold_scalp=5 * units.mm,
        sigma_brain=1 * units.mm,
        sigma_scalp=5 * units.mm,
    )

    return sbf

