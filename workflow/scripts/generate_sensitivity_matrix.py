# Script to run forward model and calculate sensitivity matrix of your probe
#%% Imports

import os
import cedalion
import cedalion.dot as dot
import cedalion.io as io
import gzip
import pickle


def generate_Adot_func(cfg_Adot, cfg_dataset, head_model, save_dir_Adot, save_dir_geo):
    #%%
    # Load recording obj from first subject/task/run
    snirf_path = os.path.join(cfg_dataset['root_dir'], f"sub-{cfg_dataset['subject'][0]}", "nirs", f"sub-{cfg_dataset['subject'][0]}_task-{cfg_dataset['task'][0]}_run-{cfg_dataset['run'][0]}_nirs.snirf")
    recordings = io.read_snirf(snirf_path)
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
    save_dir_fl = save_dir_Adot.split("sensitivity")[0]

    # calculate fluence
    print ('Calculating fluence')
    fluence_fname = os.path.join(save_dir_fl, "fluence.h5")

    if cfg_Adot['forward_model'] == "MCX":
        fwm.compute_fluence_mcx(fluence_fname)
    elif cfg_Adot['forward_model'] == "NIRFASTER":
        fwm.compute_fluence_nirfaster(fluence_fname)


    # Calculate the sensitivity matrix
    print('Calculating the sensitivity matrix')
    sensitivity_fname = os.path.join(save_dir_Adot)
    fwm.compute_sensitivity(fluence_fname, sensitivity_fname)

    # Save geometric 2d and 3d positions to sidecar file
    geo_sidecar = {
        'geo2d': rec.geo2d,
        'geo3d': rec.geo3d
        }
    file = gzip.GzipFile(save_dir_geo, 'wb')
    file.write(pickle.dumps(geo_sidecar))
    file.close()


#%%

def main():
    
    # get params
    #cfg_img_recon = snakemake.params.cfg_img_recon
    cfg_Adot = snakemake.params.cfg_Adot
    cfg_dataset = snakemake.params.cfg_dataset
    head_model = snakemake.params.head_model

    out_Adot = snakemake.output.Adot
    out_geo = snakemake.output.geometry
    
    generate_Adot_func(cfg_Adot, cfg_dataset, head_model, out_Adot, out_geo)
    
            
if __name__ == "__main__":
    main()

