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
module load GCCcore/8.3.0 GCC/8.3.0 Julia/1.5.1-linux-x86_64

# julia --machine-file ${JULIA_MACHINEFILE} --project=@. run.jl \
julia run.jl \
    --id grid_framingham1 \
    --experiment_type logistic_regression \
    --iterations 100 \
    --n_samples 4500 \
    --n_warmup 500 \
    --n_chains 1 \
    --sampler CmdStan \
    --betas 0.25 0.5 0.75 \
    --beta_weights 1.25 2.5 \
    --weights 0.0 0.5 1.0 \
    --metrics auc ll bf param_mse \
    --model_names beta weighted \
    --real_alphas 0.01 0.02 0.03 0.04 0.06 0.07 0.08 0.09 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 \
    --synth_alphas 0.0 \
    --algorithm basic \
    --sigma_p 30.0 \
    --dataset framingham \
    --epsilon 6.0 \
    --label TenYearCHD \
    --folds 5 \
    --split 1.0 \
    --path /home/dcs/csrxgb/synthetic/ \
    --distributed
    # --calibrate_beta_weight \
    # --show_progress \
