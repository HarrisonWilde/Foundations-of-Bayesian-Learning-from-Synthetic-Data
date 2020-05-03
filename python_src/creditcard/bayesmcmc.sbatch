#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=4571
#SBATCH --time=48:00:00
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

module purge
module load GCC/8.2.0-2.31.1 GCCcore/8.2.0 Python/3.7.2

srun python run.py --data 'data/raw/creditcard' --epsilons 6 \
--real_alphas 0.01 0.05 0.1 0.25 0.5 0.75 1 \
--synth_alphas 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.75 1 \
--gan 'pate' --epochs 10000 --delta 5 --teachers 10 \
--targets Class --separator ',' --split 0.6 \
--warmup 1000 --iters 6000 --chains 2 --multiplier 100 --cpu_count 28 \
--betas 0.5 --ws 1.0 --betaws 1.1 \
--n_jobs 1