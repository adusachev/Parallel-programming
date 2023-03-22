#!/bin/bash
#
#SBATCH --ntasks=8
#SBATCH --partition=RT_study
#SBATCH --nodes=1
#SBATCH --job-name=example
#SBATCH --output=out.txt
#SBATCH --error=error.txt

mpiexec ./a.out 10000
