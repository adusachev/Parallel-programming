#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include <unistd.h>
#include<pthread.h>
#include <semaphore.h>
#include<math.h>

#include <random>


#define NUM_THREADS 2
sem_t sem;
int n = 0;




double uniform_random_double(double left, double right) {

    std::random_device rd;
    std::mt19937 gen(rd());
    // std::default_random_engine rd;  // fix default random seed
 
    std::uniform_real_distribution<double> unif(left, right);
 
    double random_double = unif(rd);
    return random_double;
}

 

double integrate(int N) {
    double x_min = 0;
    double x_max = 3.14;
    double y_min = 0;
    double y_max = 1;
    double z_min = 0;
    double z_max = 3.14 / 2;

    double xi, yi, zi;
    double V_cube = (x_max - x_min) * (y_max - y_min) * (z_max - z_min);

    int n = 0;
    for (int i = 0; i < N; i++) {
        xi = uniform_random_double(x_min, x_min + x_max);
        yi = uniform_random_double(y_min, y_min + y_max);
        zi = uniform_random_double(z_min, z_min + z_max);

        if ((yi <= sin(xi)) && (zi <= xi * yi)) {
            n += 1;
        }
    }

    double ans_monte_carlo = ((double)n / N) * V_cube;
    
    return ans_monte_carlo;
}






////////////////////////////////////////////////////////////////////////////////////


void *start_func(void* param) {

	
	int *N_per_thread;
    N_per_thread = (int*) param;


    pid_t tid = gettid();

    double xi, yi, zi;
    int local = 0;
    for (int i = 0; i < *N_per_thread; i++) {
        xi = uniform_random_double(0, 3.14);
        yi = uniform_random_double(0, 1);
        zi = uniform_random_double(0, 1.57);

        if ((yi <= sin(xi)) && (zi <= xi * yi)) {
            local += 1;
        }
    }

    std::cout << "It's thread " << tid << "; local_n = " << local << std::endl;


	sem_wait(&sem);
		n += local;
	sem_post(&sem);


	// pthread_exit((void*) a);
	pthread_exit(NULL);
}



int main(int argc, char *argv[]){

  	int rc;
    int N = 1000000;
    double x_min = 0;
    double x_max = 3.14;
    double y_min = 0;
    double y_max = 1;
    double z_min = 0;
    double z_max = 3.14 / 2;
    double V_cube = (x_max - x_min) * (y_max - y_min) * (z_max - z_min);

    int N_per_thread = N / NUM_THREADS;    
	
	
	// int index[NUM_THREADS];
	pthread_t pthr[NUM_THREADS];
	struct timespec begin, end;   
	double elapsed;
    
   
	sem_init(&sem, 0, 1);
   	
	clock_gettime(CLOCK_REALTIME, &begin);
   
   
	for (int i = 0; i < NUM_THREADS; ++i){
		rc = pthread_create(&pthr[i], NULL, start_func, (void*) &N_per_thread);      
		
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
	
	elapsed = end.tv_sec - begin.tv_sec;    
	elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
	std::cout << "Time of work = " << elapsed << std::endl; 
    
    
	sem_destroy(&sem);
	
  	return 0;
}



// int main() {

//     int N = 1000;
//     double x_min = 0;
//     double x_max = 3.14;
//     double y_min = 0;
//     double y_max = 1;
//     double z_min = 0;
//     double z_max = 3.14 / 2;
//     double V_cube = (x_max - x_min) * (y_max - y_min) * (z_max - z_min);

//     int N_per_thread = N / NUM_THREADS;


//     pthread_t pthr[NUM_THREADS];
//     int n = 0;
//     void *arg;

//     sem_init(&s, 0, 1);


//     for (int i = 0; i < NUM_THREADS; i++) {
//         bool rc = pthread_create(&pthr[i], NULL, func, (void*) &param);
        
//         if (rc) {
//             printf("ERROR; return code from pthread_create() is %d \n", rc);
//         }
//     }

//     for(int i = 0; i < NUM_THREADS; i++){
//         bool rc = pthread_join(pthr[i], &arg);  // записываем по адресу &arg адрес возвращаемой переменной
//         printf("returned value from func  %d  \n", *(int*)arg);

//         if (rc) {
//             printf("ERROR; return code from pthread_join() is %d \n", rc);
//         }
//     }


    // -----------------------------------------------------------------

    // std::cout << uniform_random_double(0, 1) << std::endl; 

    // std::cout << integrate(100000) << std::endl;

    // unsigned int seed = 42;
    // for (int i = 0; i < 50; i++) {
    //     unsigned int seed = 0;
    //     std::cout << (float)rand_r(&seed) / RAND_MAX << std::endl; 
    // }

    // srand(10);
    // for (int i = 0; i < 10; i++) {
    //     std::cout << (float)rand() / RAND_MAX << std::endl; 
    // }

    // int x_k = 6;

    // float x = ((float)rand_r(&x_k) / RAND_MAX) * (i_final - i_init) + i_init;
    














    // std::random_device rd;  // Will be used to obtain a seed for the random number engine
    // std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

    // // std::default_random_engine rd;  // fix default random seed

    // // Declaring the upper and lowerbounds
    // double lower_bound = 0;
    // double upper_bound = 100;
 
    // std::uniform_real_distribution<double> unif(lower_bound,
    //                                             upper_bound);
 
    // // Getting a random double value
    // double random_double = unif(rd);
 
    // std::cout << random_double << std::endl;




//     return 0;
// }


