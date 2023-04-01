#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>


sem_t s;

void* func(void * arg) {

    sem_wait(&s);  // захватить семафор (-1)
    
    printf("Welcome! \n");
    sleep(4);
    printf("Bye!\n");
    
    sem_post(&s);  // освободить семафор (+1)

    pthread_exit(NULL);  // (?)
}



int main() {
    sem_init(&s, 0, 1);
    pthread_t o1, o2;
    
    printf("In a 1st Thread now...\n");
    
    pthread_create(&o1, NULL, func, NULL);
    
    sleep(4);
    printf("In a 2nd Thread now...\n");
    
    pthread_create(&o2, NULL, func, NULL);
    
    pthread_join(o1, NULL);
    pthread_join(o2, NULL);
    
    sem_destroy(&s);


    return 0;
}




