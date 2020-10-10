#!/bin/bash
#SBATCH --nodes=1
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

export JULIA_PROJECT=/home/dcs/csrxgb/synthetic/Project.toml
export JULIA_CMDSTAN_HOME=/home/dcs/csrxgb/cmdstan-2.24.1

module purge
module load GCC/8.3.0 Julia/1.5.1-linux-x86_64

# julia --machine-file ${JULIA_MACHINEFILE} --project=@. run.jl \
julia run.jl \
    --id neff_final \
    --experiment_type gaussian \
    --iterations 100 \
    --n_samples 4500 \
    --n_warmup 500 \
    --n_chains 1 \
    --sampler AHMC \
    --scales 0.75 1.0 \
    --betas 0.8 0.5 0.2 \
    --beta_weights 1.25 \
    --weights 1.0 0.5 0.0 \
    --metrics ll kld wass \
    --model_names beta weighted noise_aware resampled \
    --real_ns 2 4 6 8 10 13 16 19 22 25 30 35 40 50 75 100 \
    --synth_ns 120 140 160 180 200 \
    --synth_n_range 0 1 100 \
    --n_unseen 500 \
    --algorithm basic \
    --mu 0.0 \
    --sigma 1.0 \
    --mu_p 3.0 \
    --sigma_p 30.0 \
    --alpha_p 2.0 \
    --beta_p 4.0 \
    --path /home/dcs/csrxgb/synthetic/ \
    --distributed
    # --calibrate_beta_weight \
    # --show_progress \
