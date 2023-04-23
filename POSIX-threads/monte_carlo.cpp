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





struct Params {
    int thread_id;
    int points_per_thread;
};



void *start_func(void* param) {

	Params *local_pack = (Params*)param;

    int tid = local_pack->thread_id;
    int N_per_thread = local_pack->points_per_thread;

    sem_wait(&sem);
    std::cout << "Thread " << tid << "; Points per this thread " << N_per_thread << std::endl;
    sem_post(&sem);
    
    double xi, yi, zi;
    // unsigned int seed_x = tid;
    // unsigned int seed_y = tid + 1;
    // unsigned int seed_z = tid + 2;

    unsigned int seed_x = tid + 33421349820;
    unsigned int seed_y = tid + 33421349839;
    unsigned int seed_z = tid + 33421349294;



    
    // int local = 0;
    int *local_ptr = (int*) malloc(sizeof(int));

    for (int i = 0; i < N_per_thread; i++) {
        xi = ((double)rand_r(&seed_x) / RAND_MAX) * (3.14 - 0) + 0;
        yi = ((double)rand_r(&seed_y) / RAND_MAX) * (1 - 0) + 0;
        zi = ((double)rand_r(&seed_z) / RAND_MAX) * (1.57 - 0) + 0;
        
        // xi = ((double)rand() / RAND_MAX) * (3.14 - 0) + 0;
        // yi = ((double)rand() / RAND_MAX) * (1 - 0) + 0;
        // zi = ((double)rand() / RAND_MAX) * (1.57 - 0) + 0;

        // sem_wait(&sem);
        // if (i < 1) {
        //     std::cout << xi << ", " << yi << ", " << zi << std::endl;
        // }
        // sem_post(&sem);


        if ((yi <= sin(xi)) && (zi <= xi * yi)) {
            // local += 1;
            *local_ptr += 1;
        }
    }



    sem_wait(&sem);
    // std::cout << "Thread " << tid << "; local_n = " << local << std::endl;
    std::cout << "Thread " << tid << "; local_n_ptr = " << *local_ptr << std::endl;
    std::cout << "------------------------------------ " << std::endl;
    sem_post(&sem);


	// sem_wait(&sem);
	// 	n += local;
	// sem_post(&sem);


    delete local_pack;
    
	// pthread_exit(NULL);
    pthread_exit((void*) local_ptr);
}



int main(int argc, char *argv[]){

    // int NUM_THREADS = atoi(argv[1]);
    int NUM_THREADS = 4;

    void *arg;
    int ret_val;
  	int rc;
    int N = 10000000;
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

    int n_base = N / NUM_THREADS;
    int n_add = N % NUM_THREADS;

    sem_init(&sem, 0, 1);
 
    
   	/*
        Sequential calculations
    */
	clock_gettime(CLOCK_REALTIME, &begin);
    
    // double ans_master = integrate(N);
    Params* pack = new Params;
    pack->points_per_thread = N;
    pack->thread_id = 0;
    pthread_create(&pthr_master, NULL, start_func, (void*) pack);
    
    rc = pthread_join(pthr_master, &arg);
    ret_val = *(int*)arg;
    std::cout << "Value from func of the master thread is " << ret_val << std::endl;
    n += ret_val;

    // pthread_join(pthr_master, NULL);
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

        Params* pack = new Params;
        pack->thread_id = i;

        if (n_add != 0) {
            pack->points_per_thread = n_base + 1;
            n_add -= 1;
        } 
        else {
            pack->points_per_thread = n_base;
        }

		rc = pthread_create(&pthr[i], NULL, start_func, (void*) pack);
		
		if (rc){
			printf("ERROR; return code from pthread_create() is %d\n", rc);
		}
	}
   

	for (int i = 0; i < NUM_THREADS; i++) {

        rc = pthread_join(pthr[i], &arg);
        ret_val = *(int*)arg;
		std::cout << "Value from func of the thread #" << i << " is " << ret_val << std::endl;
        n += ret_val;

		// rc = pthread_join(pthr[i], NULL);
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


