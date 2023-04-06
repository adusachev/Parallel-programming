#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
//#include <math.h>
#include<iostream>

#define NUM_THREADS 5
long long int sum = 0;
sem_t sem;

void *start_func(void* id) {

	int *a = (int*) malloc(sizeof(int));
	//int a;
	int i;
	long long int local = 0;
	int val = 0;
	
	int* myid;
	//int myid;
	myid = (int*)id;
	*a = *(myid) * 10;  // (?)
	//a = *(myid) * 10;

	std::cout << "Hello World! It's thread #" << *myid << "! a = " << *a << std::endl;

	
	// sem_wait(&sem);
		for (i = 0; i < 1e7; i++){
			// sem_wait(&sem);
				local++;
				// sum ++;  // example with wrong answer (without critical section)
			// sem_post(&sem);
		}
	// sem_post(&sem);
	// sem_getvalue(&sem, &val);
	// std::cout << "Sem val outside critical = " << val << std::endl;

	sem_wait(&sem);
		sum += local;
		// sem_getvalue(&sem, &val);
		// std::cout << "Sem val inside critical = " << val << std::endl;
	sem_post(&sem);

	// sem_getvalue(&sem, &val);
	// std::cout << "Sem val outside critical = " << val << std::endl;

	//pthread_exit(NULL);
	pthread_exit((void*) a);
	//pthread_exit((void*)&a);
}



int main(int argc, char *argv[]){
	int i, param;
  	int rc;
    
  	void *arg;
    //pthread_t pthr1; 
    
    //printf("Pthr = %d, %f \n", pthr1,  pthr1);
	
	
	int index[NUM_THREADS];
	pthread_t pthr[NUM_THREADS];
	struct timespec begin, end;   
	double elapsed;
    
   
	sem_init(&sem, 0, 1);
	param = 1;
    // rc = pthread_create(&pthr, NULL, start_func, NULL);   
    // rc = pthread_create(&pthr, NULL, start_func, (void*)&param);   
    // pthread_join(pthr,  NULL);
   	
	clock_gettime(CLOCK_REALTIME, &begin);
   
   
	for(i = 1; i < NUM_THREADS; ++i){
		index[i] = i;   
		//rc = pthread_create(&pthr[i], NULL, start_func, (void*)&i);
		rc = pthread_create(&pthr[i], NULL, start_func, (void*)&index[i]);      
		if (rc){
			printf("ERROR; return code from pthread_create() is %d\n", rc);
		}
	}
   
   
    // rc = pthread_join(pthr, NULL);
    // rc = pthread_join(pthr, &arg);
    // printf("Value from thread a = %d \n", *(int*)arg);

	for (i = 1; i < NUM_THREADS; i++) {
		//rc = pthread_join(pthr[i], NULL);
		rc = pthread_join(pthr[i], &arg);
		std::cout << "Value from func of the thread #" << i << " is " << *(int*)arg << std::endl;
		//printf("Value from func of the thread # %d is %d \n", i, *(int*)arg);
		free(arg);
	}
	
	std::cout << "I am main thread. Sum = " << sum << std::endl; 
 	clock_gettime(CLOCK_REALTIME, &end);
	
	elapsed = end.tv_sec - begin.tv_sec;    
	elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
	std::cout << "Time of work = " << elapsed << std::endl; 
    
    
	sem_destroy(&sem);
	
  	return 0;
}



