# Generate Spatial Basis G matrix
#%% Imports

import os
import cedalion
import cedalion.nirs
from cedalion.physunits import units
import cedalion.dot as dot
import cedalion.io as io

import pickle
import gzip


def sbf_func(cfg_sb, head_model, Adot_path, out):

    #cfg_sb = cfg_img_recon['spatial_basis']
    # update units to config params
    cfg_sb["threshold_brain"] = units(cfg_sb["threshold_brain"])
    cfg_sb["threshold_scalp"] = units(cfg_sb["threshold_scalp"])
    cfg_sb["sigma_brain"] = units(cfg_sb["sigma_brain"])
    cfg_sb["sigma_scalp"] = units(cfg_sb["sigma_scalp"])

    head = dot.get_standard_headmodel(head_model)
    Adot = io.forward_model.load_Adot(Adot_path)

    head_ras = head.apply_transform(head.t_ijk2ras)


    sbf = dot.GaussianSpatialBasisFunctions(
        head_ras,
        Adot,
        mask_threshold = cfg_sb['mask_threshold'],
        threshold_brain = cfg_sb['threshold_brain'],
        threshold_scalp = cfg_sb['threshold_scalp'],
        sigma_brain = cfg_sb['sigma_brain'],
        sigma_scalp = cfg_sb['sigma_scalp'],
        )
    
    # NOTE: save as .h5 with cedalion func once it is updated
    # Save data to a compressed pickle file 
    print(f'   Saving to {out}')
    file = gzip.GzipFile(out, 'wb')
    file.write(pickle.dumps(sbf))
    file.close()   



def main():
    config = snakemake.config
    
    # get params
    cfg_sb = snakemake.params.cfg_sb
    head_model = snakemake.params.head_model
    Adot_path = snakemake.input.Adot
    out = snakemake.output[0]

    sbf_func(cfg_sb, head_model, Adot_path, out) # run function
            
if __name__ == "__main__":
    main()
