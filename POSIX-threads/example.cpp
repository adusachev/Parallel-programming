#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include<iostream>
#include <sys/types.h>


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



int main() {

    sem_init(&s, 0, 6);
    pthread_t o1, o2, o3, o4, o5, o6;
    
    pthread_create(&o1, NULL, func, NULL);

    pthread_create(&o2, NULL, func, NULL);

    pthread_create(&o3, NULL, func, NULL);

    pthread_create(&o4, NULL, func, NULL);

    pthread_create(&o5, NULL, func, NULL);

    pthread_create(&o6, NULL, func, NULL);

    
    pthread_join(o1, NULL);
    pthread_join(o2, NULL);
    pthread_join(o3, NULL);
    pthread_join(o4, NULL);
    pthread_join(o5, NULL);
    pthread_join(o6, NULL);



    
    sem_destroy(&s);

    return 0;
}




