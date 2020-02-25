#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4571
#SBATCH --time=03:00:00

module purge
module load GCC/8.2.0-2.31.1 GCCcore/8.2.0 Python/3.7.2

pip install --user pystan pandas tqdm seaborn numba

srun python run_k.py \
--mus 0 --sigmas 1 --scales 1.5 \
--real_alphas 0 0.05 0.1 0.25 0.5 0.75 1.0 --contaminated_alphas 0 0.01 0.02 0.03 0.04 0.05 0.075 0.1 0.25 0.5 0.75 1.0 --ks 2000 \
--priormus 0 --prioralphas 2 --priorbetas 5 --hyperpriors 5 \
--betas 0.5 --betaws 1.1 --ws 0.5 \
--warmup 1000 --iters 7000 --chains 1 --multiplier 1 --cpu_count 28 \
--ytildestep 0.01