#include <iostream>
#include <omp.h>




int main(int argc, char **argv) {
    double begin_time = omp_get_wtime(); 

    int sum = 0;
    
    omp_set_num_threads(4);  // явно задаем кол-во потоков
        
    # pragma omp parallel 
    {
        int thread_num = omp_get_thread_num();
        int cur_sum = 0;
        
        // # pragma omp for  // поделит только внешний цикл: потоки получат по 2, 2, 3, 3 итерации
        # pragma omp for collapse(2)  // поделит 10*10000=100000 итераций равномерно на 4 потока
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10000; j++) {
                cur_sum += 1;
            }
        }
        # pragma omp critical
        {
            std::cout << "Thread " << thread_num << " got sum: " << cur_sum << std::endl;
            sum += cur_sum;
        }
    }
    
    std::cout << "Result sum: "  << sum << std::endl;



    double end_time = omp_get_wtime();
    double elapsed_seconds = end_time - begin_time;
    std::cout << "Elapsed time: " << elapsed_seconds << "sec" << std::endl;
}


