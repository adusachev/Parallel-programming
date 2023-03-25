#!/bin/bash
# 
#SBATCH --ntasks=4
#SBATCH --partition=RT_build
#SBATCH --nodes=1
#SBATCH --job-name=example
#SBATCH --comment="Run mpi prog"
#SBATCH --output=out.txt
#SBATCH --error=error.txt

mpiCC -fopenmp mpi_with_openmp.cpp
mpiexec a.out
