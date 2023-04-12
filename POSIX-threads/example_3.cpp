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
    double value;
    int thread_id;
    int num;
};


void* func(void* param) {

    int *local;
    local = (int*) param;


    sem_wait(&s);  // захватить семафор (-1)

    pid_t tid = gettid();

    *local += 50;
    
    std::cout << "Value from tid " << tid << " is " << *local << std::endl;
    sleep(4);
    std::cout << "Bye " << tid << std::endl;
    
    sem_post(&s);  // освободить семафор (+1)

    pthread_exit( (void*) local);
}



int main (int argc, char *argv[]) {

    int param = 10;
    pthread_t pthr[NUM_THREADS];
    void *arg;

    sem_init(&s, 0, 1);


    for (int i = 0; i < NUM_THREADS; i++) {
        bool rc = pthread_create(&pthr[i], NULL, func, (void*) &param);
        
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d \n", rc);
        }
    }

    for(int i = 0; i < NUM_THREADS; i++){
        bool rc = pthread_join(pthr[i], &arg);  // записываем по адресу &arg адрес возвращаемой переменной
        printf("returned value from func  %d  \n", *(int*)arg);

        if (rc) {
            printf("ERROR; return code from pthread_join() is %d \n", rc);
        }
    }
 }





