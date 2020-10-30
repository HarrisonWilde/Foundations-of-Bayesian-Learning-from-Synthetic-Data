#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=10
#SBATCH --mem-per-cpu=4000
#SBATCH --time=120:00:00
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=h.wilde@warwick.ac.uk

# Create the machine file for Julia
# JULIA_MACHINEFILE=machinefile-${SLURM_JOB_ID}
# srun bash -c hostname > $JULIA_MACHINEFILE
# sed -i 's/^[[:alnum:]]*/&-ib/g' $JULIA_MACHINEFILE

export JULIA_PROJECT=/home/harrywilde/synthetic/Project.toml
# export JULIA_CMDSTAN_HOME=/home/dcs/csrxgb/cmdstan-2.24.1

# julia --machine-file ${JULIA_MACHINEFILE} --project=@. run.jl \
julia run.jl \
    --id param_mse_example \
    --experiment_type gaussian \
    --iterations 100 \
    --n_samples 4500 \
    --n_warmup 500 \
    --n_chains 1 \
    --sampler Turing \
    --scales 0.75 1.0 \
    --betas 0.8 \
    --beta_weights 1.25 \
    --weights 1.0 \
    --metrics ll kld wass param_mse \
    --model_names weighted \
    --real_ns 1 4 7 10 \
    --synth_n_range 0 1 100 \
    --real_n_range 15 5 100 \
    --n_unseen 500 \
    --algorithm basic \
    --mu 0.0 \
    --sigma 1.0 \
    --mu_p 3.0 \
    --sigma_p 30.0 \
    --alpha_p 2.0 \
    --beta_p 4.0 \
    --path /home/harrywilde/synthetic/ \
    --distributed
    # --calibrate_beta_weight \
    # --show_progress \
