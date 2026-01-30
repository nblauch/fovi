#!/bin/bash -l

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nblauch@fas.harvard.edu
#SBATCH --mem=256GB
#SBATCH --account=kempner_konkle_lab
#SBATCH --cpus-per-task 36
#SBATCH -p kempner
#SBATCH --exclude=holygpu8a19104,holygpu8a19405,holygpu8a19206

# print job number
echo "Job Number: $SLURM_JOB_ID"

echo "Node: $SLURMD_NODENAME"

cd ~/git/foveation

echo "$COMMAND"
$COMMAND
