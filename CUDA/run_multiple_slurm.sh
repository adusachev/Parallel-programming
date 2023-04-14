#!/bin/bash


for BlockSize in 4 8 16 32 64 128 256
    do
        echo "#!/bin/bash" > run.sh
        echo "#" >> run.sh
        echo "#SBATCH --ntasks 1" >> run.sh
        echo "#SBATCH --partition=titan_X" >> run.sh
        echo "#SBATCH --cpus-per-task 1" >> run.sh
        echo "#SBATCH --gres gpu:1" >> run.sh
        echo "#SBATCH --comment \"Test gpu\"" >> run.sh
        echo "#SBATCH --output=out.txt" >> run.sh
        echo "#SBATCH --error=error.txt" >> run.sh
        
        echo "./a.out" $BlockSize >> run.sh

        ## DEBUG
        # cat run.sh
        # echo "-----------------------------"

        ## RUN
        sbatch ./run.sh
        
    done
