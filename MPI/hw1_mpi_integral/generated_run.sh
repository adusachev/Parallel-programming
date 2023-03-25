#!/bin/bash
#
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --partition=RT_study
#SBATCH --nodes=1
#SBATCH --job-name=example
#SBATCH --output=out.txt
#SBATCH --error=error.txt
module add centos/8/mpi/hpcx-v2.7.0
mpiexec ./a.out 100000000
