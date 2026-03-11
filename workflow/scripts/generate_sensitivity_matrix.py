# Script to run forward model and calculate sensitivity matrix of your probe
#%% Imports

import os
import cedalion
import cedalion.nirs
import cedalion.dot as dot
import cedalion.io as io
import yaml
from cedalion.io.forward_model import load_Adot


def generate_Adot_func(cfg_Adot, probe_dir, snirf_name, head_model, save_dir):
    #%%
    # Load probe snirf
    recordings = io.read_snirf(probe_dir + snirf_name)
    rec = recordings[0]
    geo3d_meas = rec.geo3d
    meas_list = rec._measurement_lists["amp"]

    # Load head model
    head = dot.get_standard_headmodel(head_model)
    # head_ras = head.apply_transform(head.t_ijk2ras) # change between coord systems

    geo3d_snapped_ijk = head.align_and_snap_to_scalp(geo3d_meas) # optode registration, snap optodes to nearest vertex on scalp

    # Construct forward model
    fwm = dot.ForwardModel(head, geo3d_snapped_ijk, meas_list)

    #%% Run the simulation
    save_dir_fl = save_dir.split("sensitivity")[0]

    # calculate fluence
    print ('Calculating fluence')
    fluence_fname = os.path.join(save_dir_fl, "fluence.h5")

    if cfg_Adot['forward_model'] == "MCX":
        fwm.compute_fluence_mcx(fluence_fname)
    elif cfg_Adot['forward_model'] == "NIRFASTER":
        fwm.compute_fluence_nirfaster(fluence_fname)


    # Calculate the sensitivity matrix
    print('Calculating the sensitivity matrix')
    sensitivity_fname = os.path.join(save_dir)
    fwm.compute_sensitivity(fluence_fname, sensitivity_fname)
    #Adot = load_Adot(sensitivity_fname)


#%%

def main():
    config = snakemake.config
    
    # get params
    #cfg_img_recon = snakemake.params.cfg_img_recon
    cfg_Adot = snakemake.params.cfg_Adot
    probe_dir = snakemake.params.probe_dir
    snirf_name = snakemake.params.snirf_name
    head_model = snakemake.params.head_model

    out = snakemake.output[0]
    
    generate_Adot_func(cfg_Adot, probe_dir, snirf_name, head_model, out)
    
            
if __name__ == "__main__":
    main()

