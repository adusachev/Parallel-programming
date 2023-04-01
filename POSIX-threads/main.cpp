
// compile: g++ main.cpp -lpthread -lrt
// run: ./a.out


#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<pthread.h>





void* start_func(void* param) {  // принимает чистый адрес void* param
    int *local;

    local = (int*) param;

    std::cout << "hello from start_func!" << std::endl; 

    pthread_exit(NULL);  // завершение работы нити без передачи аргументов в master нить
    // pthread_exit( (void*) &arg );  // c передачей
}





int main(int argc, char *argv[]) {
        
    int param;
    int rc;
    void *arg;

    pthread_t pthr;  // идентификатор потока


    // rc = pthread_create(&pthr, NULL, start_func, NULL);

    rc = pthread_create(&pthr, NULL, start_func, (void*) &param);


    // (void*) &param - преобразование типа указателя на void

    pthread_join(pthr, NULL);  // master нить ожидает окончания работы, по умолчанию ничего не принимаем

    // pthread_join(pthr, &arg);  // записываем по адресу &arg адрес возвращаемой переменной



    return 0;
}










