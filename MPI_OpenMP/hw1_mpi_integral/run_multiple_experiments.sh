#!/bin/bash


for N in 1000 1000000 100000000
    do
        for n_proc in 1 2 3 4 5 6 7 8
            do
                echo "#!/bin/bash" > generated_run.sh
                echo "#" >> generated_run.sh
                echo "#SBATCH --ntasks="$n_proc >> generated_run.sh
                echo "#SBATCH --cpus-per-task=1" >> generated_run.sh
                echo "#SBATCH --partition=RT_study" >> generated_run.sh  # choose PARTITION (!)
                echo "#SBATCH --nodes=1" >> generated_run.sh
                echo "#SBATCH --job-name=example" >> generated_run.sh
                echo "#SBATCH --output=out.txt" >> generated_run.sh
                echo "#SBATCH --error=error.txt" >> generated_run.sh

                echo "module add centos/8/mpi/hpcx-v2.7.0"  >> generated_run.sh
                echo "mpiexec ./a.out" $N >> generated_run.sh

                ## DEBUG
                # cat generated_run.sh
                # echo "-----------------------------"

                ## RUN
                sbatch ./generated_run.sh
                sleep 20s
            done
    done

