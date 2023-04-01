
// compile: g++ main.cpp -lpthread -lrt
// run: ./a.out


#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<pthread.h>
#include<semaphore.h>  



sem_t sem;


void* start_func(void* param) {  // принимает чистый адрес void* param    
    int val;
    int *local;
    local = (int*) param;

    // ------ критическая секция ------
    sem_wait(&sem);  // захватить семафор (-1)

    std::cout << "hello from start_func!" << std::endl; 
    
    sem_post(&sem);  // освободить семафор (+1) 
    // --------------------------------

    pthread_exit(NULL);

    sem_getvalue(&sem, &val);  // проверка значения семафора
}





int main(int argc, char *argv[]) {
    
    sem_init(&sem, 0, 1);  // инициализация семафора
    
    int param;
    int rc;
    void *arg;

    pthread_t pthr;


    // rc = pthread_create(&pthr, NULL, start_func, NULL);

    rc = pthread_create(&pthr, NULL, start_func, (void*) &param);


    // (void*) &param - преобразование типа указателя на void

    pthread_join(pthr, NULL);  // master нить ожидает окончания работы, по умолчанию ничего не принимаем

    // pthread_join(pthr, &arg);  // записываем по адресу &arg адрес возвращаемой переменной

    sem_destroy(&sem);  // освобождение семафора

    return 0;
}










