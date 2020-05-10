#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --mem-per-cpu=4571
#SBATCH --time=48:00:00
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=h.wilde@warwick.ac.uk

module purge
module load GCC/8.2.0-2.31.1 GCCcore/8.2.0 Python/3.7.2

# pip install --user pystan pandas tqdm seaborn numba

python run_seb.py \
--mu 0 --sigma 1 --scale 0.25 \
--num_unseen 100 \
--priormu 1 --prioralpha 3 --priorbeta 5 --hyperprior 1 \
--weight 0.5 --beta 0.5 --betaw 1.1 \
--warmup 1000 --iters 10000 --chains 5 \
--ytildestep 0.1 \
--parallel_dgps 28 --parallel_chains 0 --iterations 10 \
--plotdir /home/dcs/csrxgb/gaussian_exp/simple/plots --outdir /home/dcs/csrxgb/gaussian_exp/simple/outputs
