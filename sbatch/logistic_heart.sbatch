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
    --id grid_heart2 \
    --experiment_type logistic_regression \
    --iterations 20 \
    --n_samples 4500 \
    --n_warmup 500 \
    --n_chains 1 \
    --sampler CmdStan \
    --betas 0.25 0.5 0.75 \
    --beta_weights 1.25 2.5 \
    --weights 0.0 0.5 1.0 \
    --metrics auc ll bf param_mse \
    --model_names beta weighted \
    --real_alphas 0.025 0.05 0.1 0.25 0.5 1.0 \
    --synth_alphas 0.0 0.01 0.02 0.03 0.04 0.05 0.075 0.1 0.15 0.2 0.4 0.6 1.0 \
    --algorithm basic \
    --sigma_p 30.0 \
    --dataset uci_heart \
    --epsilon 6.0 \
    --label target \
    --folds 5 \
    --split 1.0 \
    --path /home/dcs/csrxgb/synthetic/ \
    --distributed
    # --calibrate_beta_weight \
    # --show_progress \
