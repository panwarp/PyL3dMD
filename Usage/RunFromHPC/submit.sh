#!/bin/bash
#SBATCH --job-name=PyL3dMD
#SBATCH --partition=desktop
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --time=168:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=panwarp@msoe.edu

echo $SLURM_JOB_NODELIST > nodelist.out
echo "Slurm gave us $SLURM_CPUS_ON_NODE CPU(S) on this node."

module purge
python sample.py