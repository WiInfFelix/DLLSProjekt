#!/bin/bash
# Do not forget to select a proper partition if the default
# one is no fit for the job! You can do that either in the sbatch
# command line or here with the other settings.
#SBATCH --job-name=hello
#SBATCH --nodes=2
#SBATCH --tasks-per-node=16
#SBATCH --time=00:10:00
# Never forget that! Strange happenings ensue otherwise.
#SBATCH --export=NONE

set -e # Good Idea to stop operation on first error.

source /sw/batch/init.sh

# Load environment modules for your application here.

# Actual work starting here. You might need to call
# srun or mpirun depending on your type of application
# for proper parallel work.
# Example for a simple command (that might itself handle
# parallelisation).
echo "Hello World! I am $(hostname -s) greeting you!"
echo "Also, my current TMPDIR: $TMPDIR"

# Let's pretend our started processes are working on a
# predetermined parameter set, looking up their specific
# parameters using the set number and the process number
# inside the batch job.
export PARAMETER_SET=42
# Simplest way to run an identical command on all allocated
# cores on all allocated nodes. Use environment variables to
# tell apart the instances.
srun bash -c 'echo "process $SLURM_PROCID \
(out of $SLURM_NPROCS total) on $(hostname -s) \
parameter set $PARAMETER_SET"'
