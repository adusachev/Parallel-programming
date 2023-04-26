#include <iostream>
#include <fstream>
#include <omp.h>



// double trap_kotes(double left, double right, int n, double h) {
//     double f_0, f_n, f_i, x_cur;
//     f_0 = 4 / (1 + left * left);
//     f_n = 4 / (1 + right * right);
//     double integral = h * 0.5 * (f_0 + f_n);
//     x_cur = left + h;
//     for (int i = 0; i < n-1; i++) {
//         f_i = 4 / (1 + x_cur * x_cur);
//         integral += (f_i * h);
//         x_cur += h;
//     }
//     return integral;
// }


void write_data(double T_1, double T_p, int n_threads, std::string filename="./results.csv") {
    /*
        Write data to the end of the file filename
    */
    std::ofstream out; 
    out.open(filename, std::ios::app);
    out << T_1 << ", " << T_p << ", " << n_threads << "\n";
    out.close();
}




double integral_v1(double *f, double h, int N) {
    
    double partial_sum;
    int tid;
    double begin = omp_get_wtime();

    double integral = h * 0.5 * (f[0] + f[N]);

    #pragma omp parallel private(partial_sum, tid)
    {   
        double partial_sum = 0;  // local variable
        #pragma omp for
        for (int i = 1; i < N; i++) {
            partial_sum += f[i] * h;
        }

        tid = omp_get_thread_num();
        #pragma omp critical
        {   
            std::cout << "Thread " << tid << "; partial sum = " << partial_sum << std::endl;
            integral += partial_sum;
        }
    }

    double end = omp_get_wtime();
    double T_p = end - begin;
    std::cout << "Sum parallel: " << integral << std::endl;
    std::cout << "Time parallel: " << T_p << " sec" << std::endl;

    return T_p;
}




double integral_v2(double *f, double h, int N) {
    
    int tid;
    double begin = omp_get_wtime();
    double integral = h * 0.5 * (f[0] + f[N]);
    // std::cout << "sum default: " << integral << std::endl;

    #pragma omp parallel private(tid)
    {   
        #pragma omp for reduction(+:integral)
        for (int i = 1; i < N; i++) {
            integral += f[i] * h;
        }

        tid = omp_get_thread_num();
        #pragma omp critical
        {   
            std::cout << "Thread " << tid << "; sum: " << integral << std::endl;
        }
    }

    double end = omp_get_wtime();
    double T_p = end - begin;
    std::cout << "Sum parallel: " << integral << std::endl;
    std::cout << "Time parallel: " << T_p << " sec" << std::endl;

    return T_p;
}


double integral_v2_debug(double *f, double h, int N) {
    /*
        frunction integral_v2() with some debug info
    */
    int tid, j;
    double begin = omp_get_wtime();
    double integral = h * 0.5 * (f[0] + f[N]);
    std::cout << "sum default: " << integral << std::endl;

    #pragma omp parallel private(tid, j)
    {   
        tid = omp_get_thread_num();
        j = 1;
        #pragma omp for reduction(+:integral)
        for (int i = 1; i < N; i++) {     

            if (j == 1) {
                #pragma omp critical
                {
                    std::cout << "Thread " << tid << "; sum local init: " << integral << std::endl;
                    j++;
                }
            }

            // #pragma omp critical  // use only with small values of N
            // {
            //     std::cout << "Thread " << tid << "; i = " << i << std::endl;
            // }

            integral += f[i] * h;
        }
    }

    double end = omp_get_wtime();
    double T_p = end - begin;
    std::cout << "Sum parallel: " << integral << std::endl;
    std::cout << "Time parallel: " << T_p << " sec" << std::endl;

    return T_p;
}




int main(int argc, char *argv[]) {


    int N = (int)10;
    double h = 1e-1;

    // grid values
    double* x = new double[N+1];
    x[0] = 0;
    for (int i = 1; i < N+1; i++) {
        x[i] = x[i-1] + h;
    }
    // func values
    double* f = new double[N+1];
    for (int i = 0; i < N+1; i++) {
        f[i] = 4 / (1 + x[i] * x[i]);
    }

    /*
        sequential calculations
    */
    double begin_seq = omp_get_wtime();

    double integral = h * 0.5 * (f[0] + f[N]);

    for (int i = 1; i < N; i++) {
        integral += f[i] * h;
    }

    double end_seq = omp_get_wtime();
    double T_1 = end_seq - begin_seq;
    std::cout << "Sum sequential: " << integral << std::endl;
    std::cout << "Time sequential: " << T_1 << " sec" << std::endl;


    /*
        multithread calculations
    */
    int num_threads = 4;
    // int num_threads = atoi(argv[1]);
    
    omp_set_num_threads(num_threads);

    // three versions of implementation:
    
    // double T_p = integral_v1(f, h, N);
    double T_p = integral_v2(f, h, N);
    // double T_p = integral_v2_debug(f, h, N);

    write_data(T_1, T_p, num_threads);

    delete[] x;
    delete[] f;






    return 0;
}


