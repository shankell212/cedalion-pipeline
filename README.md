# Cedalion-Pipeline

To get a Snakemake environment working with Cedalion

	1. Go to the directory where Cedalion is cloned
	2. Edit environment_dev.yml to include 
 		- bioconda  (under _channels_) 
   		- snakemake=9.5.1 (under _dependencies_)
	3. To create your Snakemake environment with Cedalion dependencies, run:
		a. **conda env create -f environment_dev.yml -n cedalion_snakemake**
	4. Activate Snakemake:
		a. **conda activate cedalion_snakemake**
	5. Add an editable install of Cedalion to your environment
 		a. **run pip install -e .**

You now have an environment that includes both Cedalion and Snakemake
