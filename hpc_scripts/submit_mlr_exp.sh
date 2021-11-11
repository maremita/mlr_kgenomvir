#!/bin/bash

## SBATCH params are passed within the command

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip

# Install mlr_kgenomvir
# This assumes that this script is run from:
# mlr_kgenomvir/hpc_scripts
pip install --no-index ../

# Variables $PROGRAM and CONF_file are initialized with export in running script
$PROGRAM $CONF_file
