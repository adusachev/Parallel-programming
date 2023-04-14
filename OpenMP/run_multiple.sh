#!/bin/bash

for n_threads in {1..12}
    do
        echo "#!/bin/bash" > run.sh
        echo "#" >> run.sh
        echo "#SBATCH --ntasks="$n_threads >> run.sh
        echo "#SBATCH --partition=RT_build" >> run.sh  # choose PARTITION (!)
        echo "#SBATCH --nodes=1" >> run.sh
        echo "#SBATCH --job-name=example" >> run.sh
        echo "#SBATCH --output=out.txt" >> run.sh
        echo "#SBATCH --error=error.txt" >> run.sh

        echo "./a.out" $n_threads >> run.sh

        ## DEBUG
        # cat run.sh
        # echo "-----------------------------"

        ## RUN
        sbatch ./run.sh
    done
    