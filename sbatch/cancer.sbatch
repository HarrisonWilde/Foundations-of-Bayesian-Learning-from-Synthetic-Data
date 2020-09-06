#!/bin/bash
#SBATCH --nodes=11
#SBATCH --tasks-per-node=28
#SBATCH --mem-per-cpu=4571
#SBATCH --time=48:00:00
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=h.wilde@warwick.ac.uk

# Create the machine file for Julia
# JULIA_MACHINEFILE=machinefile-${SLURM_JOB_ID}
# srun bash -c hostname > $JULIA_MACHINEFILE
# sed -i 's/^[[:alnum:]]*/&-ib/g' $JULIA_MACHINEFILE

export JULIA_PROJECT=/home/dcs/csrxgb/julia_stuff/Project.toml
export JULIA_CMDSTAN_HOME=/home/dcs/csrxgb/julia_stuff/cmdstan-2.23.0

module purge
module load GCC/8.3.0 GCCcore/9.2.0 Julia/1.4.1-linux-x86_64

julia ../src/logistic_regression/run.jl \
    --path /home/dcs/csrxgb/julia_stuff \
    --dataset kag_cervical_cancer \
    --label Biopsy \
    --epsilon 6.0 \
    --iterations 100 \
    --folds 5 \
    --sampler AHMC \
    --distributed
