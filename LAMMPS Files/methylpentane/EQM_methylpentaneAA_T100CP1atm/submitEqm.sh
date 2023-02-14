#!/bin/bash
#SBATCH --job-name=eqm
#SBATCH --partition=medium
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=28
#SBATCH --time=24:00:00
#SBATCH --export=ALL

echo $SLURM_JOB_NODELIST > nodelist.out

module purge
LMP=/home/ppanwar/lammps-21Jul20/src/lmp_mpi
module load openmpi/4.1.1-gcc-8.4.1

export OMP_NUM_THREADS=1
NSLOTS=$(($SLURM_NNODES*$SLURM_NTASKS_PER_NODE))
mpirun -mca btl_tcp_if_include ib0 -np $NSLOTS $LMP -log log.lammps -in in.eqm > realtimelog.txt