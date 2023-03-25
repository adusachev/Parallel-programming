#include <iostream>
#include <omp.h>

int main(int argc, char** argv) {
	


    std::cout << "-------------------------------------------" << std::endl;
    
    int num_procs = omp_get_num_procs();
    std::cout << "Number of available processors: " << num_procs << std::endl;
    
    omp_set_num_threads(num_procs);
    int num_threads = omp_get_num_threads();
    std::cout << "Number of working threads (main part): " << num_threads << std::endl;
    
    int thread_id = omp_get_thread_num();
    std::cout << "Thread id (main part): " << thread_id << std::endl;
    
    std::cout << "-------------------------------------------" << std::endl;

    // входим в параллельную область и указываем, 
    // что переменная thread_id будет локальной для каждого потока
    #pragma omp parallel private(thread_id)
	{
		thread_id = omp_get_thread_num();  // записываем в эту переменную номер каждого потока
		#pragma omp critical
		{
		    std::cout << "Hello world from thread " << thread_id << std::endl;
		}
		
		if (thread_id == 0) {
			num_threads = omp_get_num_threads();

			std::cout << "Number of threads " << num_threads << std::endl;
		}
	}
}












