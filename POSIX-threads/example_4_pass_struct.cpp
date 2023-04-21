#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <sys/types.h>
#include<iostream>
#include<random>


#define NUM_THREADS 3
sem_t s;


struct Params {
    int thread_id;
    int num;
};


void* func(void* mypack) {

    Params *local = (Params*)mypack;

    sem_wait(&s);

    int tid = local->thread_id;
    int val = local->num;
    
    std::cout << "tid: " << tid << std::endl;
    std::cout << "num: " << val << std::endl;
    
    sem_post(&s);


    delete local;  // free memory
    pthread_exit(NULL);

}



int main (int argc, char *argv[]) {

    struct timespec begin, end; 
    double elapsed;
    int param = 10;
    pthread_t pthr[NUM_THREADS];


    sem_init(&s, 0, 1);

    clock_gettime(CLOCK_REALTIME, &begin);

    // struct Params *mypack;

    for (int i = 0; i < NUM_THREADS; i++) {
        Params* mypack = new Params;  // allocate memory
        mypack->thread_id = i;

        if (i == 0) {
            mypack->num = 10;
        }
        else {
            mypack->num = 22;
        }

        pthread_create(&pthr[i], NULL, func, (void*) mypack);
    }

    for(int i = 0; i < NUM_THREADS; i++) {
        pthread_join(pthr[i], NULL);
    }


    clock_gettime(CLOCK_REALTIME, &end);
    elapsed = end.tv_sec - begin.tv_sec;    
	elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
	std::cout << "Time of work = " << elapsed << std::endl; 

    return 0;
 }





