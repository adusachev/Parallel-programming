
#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include<mpi.h>



void write_data(int N, double T_1, double T_p, int n_proc, std::string filename="./results.csv") {
    /*
    Write data to the end of the file filename
    */
    std::ofstream out; 
    out.open(filename, std::ios::app);
    out << N << ", " << T_1 << ", " << T_p << ", " << n_proc << "\n";
    out.close();
}


int N_from_cli(int argc, char *argv[]) {
    /*
    Returns an integer passed through the command line to the main function
    (e.g.: ./a.out 1000)
    */
    assert (argc == 2);
    int val;
    std::istringstream iss( argv[1] );
    iss >> val;

    return val;
}




double trap_kotes(double left, double right, int n, double h) {
    double f_0, f_n, f_i, x_cur;
    f_0 = 4 / (1 + left * left);
    f_n = 4 / (1 + right * right);
    double integral = h * 0.5 * (f_0 + f_n);
    x_cur = left + h;
    for (int i = 0; i < n-1; i++) {
        f_i = 4 / (1 + x_cur * x_cur);
        integral += (f_i * h);
        x_cur += h;
    }

    return integral;
}



int main(int argc, char *argv[]) {
    int i;
    int pid, n_proc;
    double sum, partial_sum;
    int n_single, n_single_last, last_index;
    double begin_time, end_time, T_1, T_p, begin_seq, end_seq, global_begin_time, global_end_time, T_global;
    double h;
    double left, right, last_left, last_right;
    double f_cur, f_next;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Status Status;


    int N = N_from_cli(argc, argv);

    double a = 0;
    double b = 1;
    h = (double)(b - a) / N;
   

    if (pid != n_proc-1) {
        n_single = (N / n_proc);
    } 
    else {
        n_single = (N / n_proc) + (N % n_proc);
    }
    

    // master process
    if (pid == 0) {
        /*
            sequential calculations
        */
        begin_seq = MPI_Wtime();
        double sum_seq = trap_kotes(a, b, N, h);
        end_seq = MPI_Wtime();
        T_1 = end_seq - begin_seq;
        std::cout << " ------------------------- \n Sum sequential = " << sum_seq 
                    << "\n Time sequential: " << T_1 << " sec" << 
                    "\n -------------------------" << std::endl;
        

        /*
            parallel calculations
        */
        if (n_proc > 1) {
            begin_time = MPI_Wtime();

            // send borders to slaves
            for (i = 1; i < (n_proc-1); i++) {
                left = a + i * (h*n_single);
                right = left + (h*n_single);
                MPI_Send(&left, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
                MPI_Send(&right, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
            }
            // send last part borders, which may have different size
            last_left = a + (n_proc - 1) * (h*n_single);
            last_right = b;
            MPI_Send(&last_left, 1, MPI_DOUBLE, (n_proc-1), (n_proc-1), MPI_COMM_WORLD);
            MPI_Send(&last_right, 1, MPI_DOUBLE, (n_proc-1), (n_proc-1), MPI_COMM_WORLD);

            // master processing its part of array
            left = a;
            right = left + (h*n_single);
            int n_single = (right - left) / h;
            sum = trap_kotes(left, right, n_single, h);
            std::cout << "pid: " << pid << " --> " << "partial sum = " << sum << std::endl;


            // master recieve result from other processes and combine all results
            for (i = 1; i < n_proc; i++) {
                MPI_Recv(&partial_sum, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &Status);
                sum += partial_sum; 
            }

            end_time = MPI_Wtime();

            // final result and time
            T_p = end_time - begin_time;
            std::cout << " ------------------------- \n Sum parallel = " << sum 
                    << "\n Time parallel: " << T_p << " sec" << 
                    "\n -------------------------" << std::endl;
  
        }
        else if (n_proc == 1) {
            T_p = T_1;
        }
        // save results
        write_data(N, T_1, T_p, n_proc);
    }



    if (pid != 0) {        
        // recieve data from master process
        MPI_Recv(&left, 1, MPI_DOUBLE, 0, pid, MPI_COMM_WORLD, &Status);
        MPI_Recv(&right, 1, MPI_DOUBLE, 0, pid, MPI_COMM_WORLD, &Status);
        
        // perform calculations
        double x_cur = left;
        double x_next = x_cur + h;
        double partial_sum = 0;

        partial_sum = trap_kotes(left, right, n_single, h);
        std::cout << "pid: " << pid << " --> partial sum = " << partial_sum << std::endl;
        
        // send calc results back to master
        MPI_Send(&partial_sum, 1, MPI_DOUBLE, 0, pid, MPI_COMM_WORLD);
    }


    
    MPI_Finalize();

    return 0;
}

