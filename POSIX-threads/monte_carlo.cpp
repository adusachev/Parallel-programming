#include<stdio.h>
#include<iostream>
#include<fstream>
#include<stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
#include<pthread.h>
#include <semaphore.h>
#include<math.h>

#include <random>

#define gettid() syscall(SYS_gettid)

// #define NUM_THREADS 2
sem_t sem;
int n = 0;



void write_data(double T_1, double T_p, int n_threads, std::string filename="./results.csv") {
    /*
        Write data to the end of the file filename
    */
    std::ofstream out; 
    out.open(filename, std::ios::app);
    out << T_1 << ", " << T_p << ", " << n_threads << "\n";
    out.close();
}


 

// double integrate(int N) {
//     double x_min = 0;
//     double x_max = 3.14;
//     double y_min = 0;
//     double y_max = 1;
//     double z_min = 0;
//     double z_max = 3.14 / 2;

//     double xi, yi, zi;
//     double V_cube = (x_max - x_min) * (y_max - y_min) * (z_max - z_min);

//     int n_master = 0;
//     for (int i = 0; i < N; i++) {
//         xi = ((float)rand() / RAND_MAX) * (3.14 - 0) + 0;
//         yi = ((float)rand() / RAND_MAX) * (1 - 0) + 0;
//         zi = ((float)rand() / RAND_MAX) * (1.57 - 0) + 0;

//         if ((yi <= sin(xi)) && (zi <= xi * yi)) {
//             n_master += 1;
//         }
//     }

//     double ans_monte_carlo = ((double)n_master / N) * V_cube;
    
//     return ans_monte_carlo;
// }







void *start_func(void* param) {

	

	int *N_per_thread;
    N_per_thread = (int*) param;

    pid_t tid = gettid();
    
    sem_wait(&sem);
    double xi, yi, zi;
    unsigned int seed_x = (unsigned int)tid * 33421;
    unsigned int seed_y = (unsigned int)tid * 55893;
    unsigned int seed_z = (unsigned int)tid * 14901;

    // xi = ((double)rand_r(&seed_x) / RAND_MAX) * (3.14 - 0) + 0;
    // yi = ((double)rand_r(&seed_y) / RAND_MAX) * (1 - 0) + 0;
    // zi = ((double)rand_r(&seed_z) / RAND_MAX) * (1.57 - 0) + 0;

    // std::cout << "It's one thread " << std::endl;

    // std::cout << seed_x << " " << seed_y << " " << seed_z << std::endl;
    std::cout << "Thread " << tid << "; Points per this thread " << *N_per_thread << std::endl;
    sem_post(&sem);


    
    int local = 0;
    for (int i = 0; i < *N_per_thread; i++) {
        xi = ((float)rand_r(&seed_x) / RAND_MAX) * (3.14 - 0) + 0;
        yi = ((float)rand_r(&seed_y) / RAND_MAX) * (1 - 0) + 0;
        zi = ((float)rand_r(&seed_z) / RAND_MAX) * (1.57 - 0) + 0;
        
        // xi = ((float)rand() / RAND_MAX) * (3.14 - 0) + 0;
        // yi = ((float)rand() / RAND_MAX) * (1 - 0) + 0;
        // zi = ((float)rand() / RAND_MAX) * (1.57 - 0) + 0;

        // sem_wait(&sem);
        // if (i < 1) {
        //     std::cout << xi << ", " << yi << ", " << zi << std::endl;
        // }
        // sem_post(&sem);


        if ((yi <= sin(xi)) && (zi <= xi * yi)) {
            local += 1;
        }
    }

    std::cout << "Thread " << tid << "; local_n = " << local << std::endl;
    std::cout << "------------------------------------ " << std::endl;



	sem_wait(&sem);
		n += local;
	sem_post(&sem);


	// pthread_exit((void*) a);
	pthread_exit(NULL);
}



int main(int argc, char *argv[]){

    int NUM_THREADS = atoi(argv[1]);

  	int rc;
    int N = 100000000;
    double x_min = 0;
    double x_max = 3.14;
    double y_min = 0;
    double y_max = 1;
    double z_min = 0;
    double z_max = 3.14 / 2;
    double V_cube = (x_max - x_min) * (y_max - y_min) * (z_max - z_min);
	
	
	pthread_t pthr[NUM_THREADS];
    pthread_t pthr_master;
	struct timespec begin, end;   
	double elapsed, T1, Tp;

    int num_per_thread[NUM_THREADS];
    int n_base = N / NUM_THREADS;
    int n_add = N % NUM_THREADS;

    sem_init(&sem, 0, 1);
 
    
   	/*
        Sequential calculations
    */
	clock_gettime(CLOCK_REALTIME, &begin);
    
    // double ans_master = integrate(N);
    pthread_create(&pthr_master, NULL, start_func, (void*) &N);
    
    pthread_join(pthr_master, NULL);
    double ans_master = ((double)n / N) * V_cube;

    clock_gettime(CLOCK_REALTIME, &end);
	
	T1 = end.tv_sec - begin.tv_sec;    
	T1 += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
	std::cout << "Time of work sequential  = " << T1 << " sec" << std::endl; 
    std::cout << "Answer sequential: integral = " << ans_master << std::endl; 


    /*
        Multithread calculations
    */
    n = 0;

    
    clock_gettime(CLOCK_REALTIME, &begin);
   
	for (int i = 0; i < NUM_THREADS; ++i) {
        if (n_add != 0) {
            num_per_thread[i] = n_base + 1;
            n_add -= 1;
        } 
        else {
            num_per_thread[i] = n_base;
        }
		rc = pthread_create(&pthr[i], NULL, start_func, (void*) &num_per_thread[i]);      
		
		if (rc){
			printf("ERROR; return code from pthread_create() is %d\n", rc);
		}
	}
   

	for (int i = 0; i < NUM_THREADS; i++) {
		rc = pthread_join(pthr[i], NULL);
		// rc = pthread_join(pthr[i], &arg);
		// std::cout << "Value from func of the thread #" << i << " is " << *(int*)arg << std::endl;
	}
	
	std::cout << "I am main thread. n = " << n << std::endl; 
	std::cout << "Answer: integral = " << ((double)n / N) * V_cube << std::endl; 



 	clock_gettime(CLOCK_REALTIME, &end);
	
	Tp = end.tv_sec - begin.tv_sec;    
	Tp += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
	std::cout << "Time of work = " << Tp << " sec" << std::endl; 
    
	sem_destroy(&sem);


    write_data(T1, Tp, NUM_THREADS);




	
  	return 0;
}



