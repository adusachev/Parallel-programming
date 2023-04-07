#!/bin/bash


for BlockSize in 4 8 16 32
    do
        echo "#!/bin/bash" > generated_run.sh
        echo "#" >> generated_run.sh
        echo "#SBATCH --ntasks 1" >> generated_run.sh
        echo "#SBATCH --partition=titan_X" >> generated_run.sh
        echo "#SBATCH --gres gpu:1" >> generated_run.sh
        echo "#SBATCH --comment "Test gpu"" >> generated_run.sh
        echo "#SBATCH --output=out.txt" >> generated_run.sh
        echo "#SBATCH --error=error.txt" >> generated_run.sh

        echo "./a.out" $BlockSize >> generated_run.sh

        ## DEBUG
        cat generated_run.sh
        echo "-----------------------------"

        ## RUN
        # sbatch ./generated_run.sh
        
    done
