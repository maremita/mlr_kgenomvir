#!/bin/bash

## SBATCH params are passed within the command

module load StdEnv/2020
module load python/3.7
module load imkl/2020.1.217
module load java/13.0.2

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip

# Install mlr_kgenomvir
# This assumes that this script is run from:
# mlr_kgenomvir/hpc_scripts

# if any package is not available within the cluster environment
# download it using pip before running the jobs
# pip download --no-deps phylodm 

pip install --no-index ../../phylodm-1.3.1.tar.gz

pip install --no-index ../

# Variables $PROGRAM and CONF_file are initialized with export in running script
$PROGRAM $CONF_file
