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
    double value1;
    double value2;
    double value3;
    int thread_id;
    int num;
};


void* func(void* mypack) {

    // Params *local_pack;
    // local_pack = (Params*) mypack;

    int *local;
    local = (int*) mypack;


    sem_wait(&s);  // захватить семафор (-1)

    pid_t tid = gettid();

    // *local += 50;
    
    // std::cout << "Value1 from tid " << local_pack->thread_id << " is " << local_pack->value1 << std::endl;
    // std::cout << "Value2 from tid " << local_pack->thread_id << " is " << local_pack->value2 << std::endl;
    // std::cout << "Value3 from tid " << local_pack->thread_id << " is " << local_pack->value3 << std::endl;

    std::cout << "Value1 from tid " << tid << " is " << *local << std::endl;
    std::cout << "Value2 from tid " << tid << " is " << *local << std::endl;
    std::cout << "Value3 from tid " << tid << " is " << *local << std::endl;

    // sleep(4);
    std::cout << "Bye " << tid << std::endl;
    
    sem_post(&s);  // освободить семафор (+1)

    // pthread_exit( (void*) local);
    pthread_exit(NULL);

}



int main (int argc, char *argv[]) {

    struct timespec begin, end; 
    double elapsed;
    int param = 10;
    Params mypack;

    mypack.num = 100;
    mypack.value1 = 3.14;
    mypack.value2 = 3.15;
    mypack.value3 = 3.16;



    int index[NUM_THREADS];
    pthread_t pthr[NUM_THREADS];
    void *arg;

    sem_init(&s, 0, 1);

    clock_gettime(CLOCK_REALTIME, &begin);


    for (int i = 0; i < NUM_THREADS; i++) {
        // mypack.thread_id = i;
        // bool rc = pthread_create(&pthr[i], NULL, func, (void*) &mypack);

        bool rc = pthread_create(&pthr[i], NULL, func, (void*) &param);
        
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d \n", rc);
        }
    }

    for(int i = 0; i < NUM_THREADS; i++) {
        bool rc = pthread_join(pthr[i], NULL);
        // bool rc = pthread_join(pthr[i], &arg);  // записываем по адресу &arg адрес возвращаемой переменной
        // printf("returned value from func  %d  \n", *(int*)arg);

        if (rc) {
            printf("ERROR; return code from pthread_join() is %d \n", rc);
        }
    }


    clock_gettime(CLOCK_REALTIME, &end);
    elapsed = end.tv_sec - begin.tv_sec;    
	elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
	std::cout << "Time of work = " << elapsed << std::endl; 


 }





