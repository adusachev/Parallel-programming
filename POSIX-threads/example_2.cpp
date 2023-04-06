#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include<iostream>
#include <sys/types.h>


#define NUM_THREADS 2
sem_t s;




void* func(void * arg) {

    sem_wait(&s);  // захватить семафор (-1)

    pid_t tid = gettid();
    
    std::cout << "Welcome to " << tid << std::endl;
    sleep(4);
    std::cout << "Bye " << tid << std::endl;
    
    sem_post(&s);  // освободить семафор (+1)

    pthread_exit(NULL);  // (?)
}



int main (int argc, char *argv[]) {
     pthread_t pthr[NUM_THREADS];
     void *arg;
     sem_init(&s, 0, 1);

    
     for (int i = 0; i < NUM_THREADS; i++) {
        bool rc = pthread_create(&pthr[i], NULL, func, NULL);
        if (rc) {
              printf("ERROR; return code from pthread_create() is %d \n", rc);
        }
     } 
    
     for(int i = 0; i < NUM_THREADS; i++){
        bool rc = pthread_join(pthr[i], &arg);  // !!! обработка аргументов из нити
        // printf("value from func  %d  \n", *(int*)arg);      // в качестве примера   
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d \n", rc);
        }
     }
 }





