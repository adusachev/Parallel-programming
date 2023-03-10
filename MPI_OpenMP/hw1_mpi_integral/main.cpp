
#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include <fstream>
#include <string>
#include<mpi.h>


void write_data(int N, double T_1, double T_p, int n_proc, std::string filename="./results.csv") {
    /*
    Записывает данные в конец файла filename
    */
    std::ofstream out; 
    out.open(filename, std::ios::app);
    out << N << ", " << T_1 << ", " << T_p << ", " << n_proc << "\n";
    out.close();
}



int main(int argc, char *argv[]) {
    int i;
    int pid, n_proc;
    double sum, partial_sum;
    int n_single, n_single_last, last_index;
    double begin_time, end_time, T_1, T_p, begin_seq, end_seq;
    double h;
    MPI_Status Status;  
    int N = 1000;
    double f[N+1];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    // begin_time = MPI_Wtime();

    // master process
    if (pid == 0) {

        // create array
        double a = 0;
        double b = 1;
        h = (double)b / N;
        // grid values
        double x[N+1];
        x[0] = 0;
        for (int i = 1; i < N+1; i++) {
            x[i] = x[i-1] + h;
        }
        // func values
        for (int i = 0; i < N+1; i++) {
            f[i] = 4 / (1 + x[i] * x[i]);
        }


        /*
            sequential calculations
        */
        begin_seq = MPI_Wtime();
        sum = 0;
        for (i = 0; i < N; i++) {
            sum += (f[i] + f[i+1]) * h * 0.5;
        }
        end_seq = MPI_Wtime();
        T_1 = end_seq - begin_seq;
        std::cout << "Sum sequential: " << sum << std::endl;
        std::cout << "Time sequential: " << T_1 << " sec" << std::endl;


        /*
            parallel calculations
        */
        if (n_proc > 1) {
            begin_time = MPI_Wtime();


            n_single = N / n_proc;  // size of subarray for each process
            n_single_last = n_single + (N % n_proc);

            // send parts of array to other processes
            for (i = 1; i < (n_proc-1); i++) {
                MPI_Send(&h, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
                MPI_Send(&n_single, 1, MPI_INT, i, i, MPI_COMM_WORLD);
                MPI_Send(&f[i * n_single], n_single, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
            }
            // send last part of array, which may have different size
            last_index = (n_proc-1);
            MPI_Send(&h, 1, MPI_DOUBLE, last_index, last_index, MPI_COMM_WORLD);
            MPI_Send(&n_single_last, 1, MPI_INT, last_index, last_index, MPI_COMM_WORLD);
            MPI_Send(&f[N - n_single_last], n_single_last, MPI_DOUBLE, last_index, last_index, MPI_COMM_WORLD);


            // master processing its part of array
            sum = 0;
            for (i = 0; i < n_single; i++) {
                sum += (f[i] + f[i+1]) * h * 0.5;
            }
            std::cout << "pid " << pid << "; " << "partial_sum: " << sum << std::endl;

            // master recieve result from other processes and combine all results
            for (i = 1; i < n_proc; i++) {
                MPI_Recv(&partial_sum, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &Status);
                sum += partial_sum; 
            }


            // final result
            std::cout << " -------------------- \n integral = " << sum 
                    << "\n --------------------" << std::endl;


            // print time
            end_time = MPI_Wtime();
            T_p = end_time - begin_time;
            std::cout << "Time parallel: " << T_p << " sec" << std::endl;

            // save results
            write_data(N, T_1, T_p, n_proc);
        }
    }



    if (pid != 0) {        
        // recieve data from master process
        MPI_Recv(&h, 1, MPI_DOUBLE, 0, pid, MPI_COMM_WORLD, &Status);
        MPI_Recv(&n_single, 1, MPI_INT, 0, pid, MPI_COMM_WORLD, &Status);
        MPI_Recv(&f[0], n_single, MPI_DOUBLE, 0, pid, MPI_COMM_WORLD, &Status);

        // perform some calculations
        std::cout << "pid " << pid << "; ";
        double partial_sum = 0;
        for (i = 0; i < n_single; i++) {
            // partial_sum += array[i];
            partial_sum += (f[i] + f[i+1]) * h * 0.5;
        }
        std::cout << "partial_sum: " << partial_sum << std::endl;
               
        // send calc results back to master
        MPI_Send(&partial_sum, 1, MPI_DOUBLE, 0, pid, MPI_COMM_WORLD);
    }





    MPI_Finalize();


    return 0;
}



