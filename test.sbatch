#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=28
#SBATCH --mem-per-cpu=4571
#SBATCH --time=01:00:00
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

# Create the machine file for Julia
JULIA_MACHINEFILE=machinefile-${SLURM_JOB_ID}
srun bash -c hostname > $JULIA_MACHINEFILE
sed -i 's/^[[:alnum:]]*/&-ib/g' $JULIA_MACHINEFILE

module purge
module load Julia/1.4.1-linux-x86_64

julia --project=@. --machine-file ${JULIA_MACHINEFILE} src/creditcard/cluster_test.jl 2000 2000
