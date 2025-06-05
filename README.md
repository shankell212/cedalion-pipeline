# Cedalion-Pipeline

To get snakemake env working w/ cedalion

	1. Go to where cedalion is cloned
	2. Change environment_dev.yml to include bioconda and snakemake
	3. Create your snakemake environment with cedalion dependencies
		a. conda env create -f environment_dev.yml -n cedalion_snakemake
	4. Activate snakemake 
		a. conda activate cedalion_snakemake
	5. run pip install -e .
    a. this installs cedalion into your snakemake env
