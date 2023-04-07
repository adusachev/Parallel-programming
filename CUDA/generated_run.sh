#!/bin/bash
#
#SBATCH --ntasks 1
#SBATCH --partition=titan_X
#SBATCH --gres gpu:1
#SBATCH --comment Test gpu
#SBATCH --output=out.txt
#SBATCH --error=error.txt
./a.out 32
