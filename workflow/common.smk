# common.smk
#
# Rules shared between the default `Snakefile` (preprocess -> HRF estimation ->
# image recon -> group average) and `Snakefile_reconfirst` (preprocess -> image
# recon -> HRF estimation -> group average). Both entry points `include:` this
# file so `preprocess`, `generate_sensitivity`, and `generate_sbf` are defined
# exactly once instead of duplicated.
#
# This file relies on names already being defined in the including Snakefile
# before its `include:` statement is reached: SNAKEFILE_DIR, ROOT, config, and
# root_join(). `include:` is a literal textual include -- it runs in the same
# namespace as the including file, so those names are visible here as long as
# the include comes after they're defined.

rule preprocess:
    input:
        # raw .snirf and events.tsv
        snirf = root_join("sub-{subject}", "nirs", "sub-{subject}_task-{task}_run-{run}_nirs.snirf"),
        events = root_join("sub-{subject}", "nirs", "sub-{subject}_task-{task}_run-{run}_events.tsv"),
        module1 = SNAKEFILE_DIR / "scripts/modules/module_preprocess.py",
        module2 = SNAKEFILE_DIR / "scripts/modules/module_plot_DQR.py"
    output:
        snirf = (f"{SAVE_DIR}/Outputs/preprocessed_data/sub-{{subject}}/sub-{{subject}}_task-{{task}}_run-{{run}}_nirs_preprocessed.snirf"), # preprocessed snirf
        sidecar = (f"{SAVE_DIR}/Outputs/preprocessed_data/sub-{{subject}}/sub-{{subject}}_task-{{task}}_run-{{run}}_nirs_dataquality.nc"), # data qual sidecar
    params:
        cfg_preprocess = config['preprocess'],
        root_dir = ROOT,
        derivatives_subfolder = config['dataset']['derivatives_subfolder'],
        stim_lst = config['hrf_estimation']['stim_lst'],
    script:
        SNAKEFILE_DIR / "scripts/preprocess.py"


rule generate_sensitivity:
    input:
        in1 = []
    output:
        Adot = protected(root_join('derivatives', 'cedalion', 'forward', config['image_recon']['generate_sensitivity']['sub_folder'], 'sensitivity.nc')),
    resources:
        gpu=1
    params:
        root_dir = config['dataset']["root_dir"],
        head_model = config['image_recon']['generate_sensitivity']['head_model'],
        cfg_Adot = config['image_recon']['generate_sensitivity']
    script:
        SNAKEFILE_DIR / "scripts/generate_sensitivity_matrix.py"

if config['image_recon']['spatial_basis']['enable']: # run generate sbf rule if enable is true
    rule generate_sbf:
        input:
            Adot = ancient(rules.generate_sensitivity.output)
        output: # CHANGE file ext when save and load sbf in ced updated
            root_join('derivatives', 'cedalion', 'forward', config['image_recon']['generate_sensitivity']['sub_folder'], 'sbf.nc')
        params:
            cfg_sb = config['image_recon']['spatial_basis'],
            head_model = config['image_recon']['generate_sensitivity']['head_model']
        script:
            SNAKEFILE_DIR / "scripts/generate_sbf.py"  #FIXME: this file is not finished
