#!/bin/bash
#
#SBATCH --ntasks 1
#SBATCH --partition=titan_X
#SBATCH --cpus-per-task 1
#SBATCH --gres gpu:1
#SBATCH --comment "Test gpu"
#SBATCH --output=out.txt
#SBATCH --error=error.txt
nvcc MatrixMul.cu
./a.out 256
