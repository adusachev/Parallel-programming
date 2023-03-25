#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<mpi.h>  




int main(int argc, char *argv[]) {
    int i;
    int pid, n_proc;
    int sum, partial_sum;
    int N, n_single, n_single_last, last_index;
    
    MPI_Status Status;  


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    
    N = 12;
    int* array = new int[N];


    

    // master process
    if (pid == 0) {
        double begin_time, end_time, total_time;

        // create array
        for (i = 0; i < N; i++) {
            array[i] = i;
        }
        // std::cout << array[N-1] << std::endl;

        /*
            sequential calculations
        */
        begin_time = MPI_Wtime();
        sum = 0;
        for (i = 0; i < N; i++) {
            sum += array[i];
        }
        end_time = MPI_Wtime();
        total_time = end_time - begin_time;
        std::cout << "Sum sequential: " << sum << std::endl;
        std::cout << "Time sequential: " << total_time << " sec" << std::endl;


        /*
            parallel calculations
        */
        begin_time = MPI_Wtime();

        n_single = N / n_proc;  // size of subarray for each process

        // send parts of array to other processes
        for (i = 1; i < (n_proc-1); i++) {
            MPI_Send(&n_single, 1, MPI_INT, i, i, MPI_COMM_WORLD);
            MPI_Send(&array[i * n_single], n_single, MPI_INT, i, i, MPI_COMM_WORLD);
        }
        // send last part of array, which may have different size
        last_index = (n_proc-1);
        n_single_last = n_single + (N % n_proc);
        MPI_Send(&n_single_last, 1, MPI_INT, last_index, last_index, MPI_COMM_WORLD);
        MPI_Send(&array[N - n_single_last], n_single_last, MPI_INT, last_index, last_index, MPI_COMM_WORLD);


        // master processing its part of array
        sum = 0;
        for (i = 0; i < n_single; i++) {
            sum += array[i];
        }
        std::cout << "pid " << pid << "; " << "partial_sum: " << sum << std::endl;

        // master recieve result from other processes and combine all results
        for (i = 1; i < n_proc; i++) {
            MPI_Recv(&partial_sum, 1, MPI_INT, i, i, MPI_COMM_WORLD, &Status);
            sum += partial_sum; 
        }

        // final result
        std::cout << " -------------------- \n Sum parallel = " << sum 
                << "\n --------------------" << std::endl;

        // print time
        end_time = MPI_Wtime();
        total_time = end_time - begin_time;
        std::cout << "Time parallel: " << total_time << " sec" << std::endl;
    }


    if (pid != 0) {        
        // recieve data from master process
        MPI_Recv(&n_single, 1, MPI_INT, 0, pid, MPI_COMM_WORLD, &Status);
        MPI_Recv(&array[0], n_single, MPI_INT, 0, pid, MPI_COMM_WORLD, &Status);

        // perform some calculations
        std::cout << "pid " << pid << "; ";
        int partial_sum = 0;
        for (i = 0; i < n_single; i++) {
            partial_sum += array[i];
        }
        std::cout << "partial_sum: " << partial_sum << std::endl;
               
        // send calc results back to master
        MPI_Send(&partial_sum, 1, MPI_INT, 0, pid, MPI_COMM_WORLD);
    }



    delete[] array;    

    MPI_Finalize();


    return 0;
}

