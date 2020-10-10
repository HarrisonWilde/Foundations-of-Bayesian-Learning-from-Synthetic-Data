#!/bin/bash
#SBATCH --nodes=8
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
    --id test2 \
    --experiment_type gaussian \
    --iterations 100 \
    --n_samples 10000 \
    --n_warmup 1000 \
    --n_chains 1 \
    --sampler AHMC \
    --scales 0.75 1.0 1.25 \
    --betas 0.2 0.4 0.6 0.8 \
    --beta_weights 1.25 2.5 \
    --weights 0.0 0.25 0.5 0.75 1.0 \
    --metrics ll kld wass \
    --model_names beta weighted noise_aware \
    --real_ns 5 10 25 50 100 \
    --synth_ns 0 5 10 15 20 25 30 40 50 75 100 200 \
    --n_unseen 100 \
    --algorithm basic \
    --mu 0.0 \
    --sigma 1.0 \
    --mu_p 1.0 \
    --sigma_p 10.0 \
    --alpha_p 2.0 \
    --beta_p 4.0 \
    --path /home/dcs/csrxgb/synthetic/ \
    --distributed
    # --calibrate_beta_weight \
    # --show_progress \
