#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
//#include <math.h>

#define NUM_THREADS     4
long long int sum = 0;
sem_t sem;

void *start_func(void* id)
{
  int* myid;
	myid = (int*)id;
  //*a = *(myid) * 10;

  int a, b;
  a = 10;
  b = 20;
  a = a + b;

  a = *(myid) * 10;

   
  //printf("Hello! a = %d \n", a);
  printf("Hello World! It's thread # %d! a = %d \n", *myid, a);   //, *a);	

  pthread_exit(NULL);
}



int main (int argc, char *argv[]){
	int i, param;
  int rc;
  double elapsed;
  void *arg;
    
  pthread_t pthr; 
  printf("Pthr = %d \n", pthr);
		
  param = 1;

  //rc = pthread_create(&pthr, NULL, start_func, NULL);   
    
  rc = pthread_create(&pthr, NULL, start_func, (void*)&param);   
  printf("rc = %d \n", rc);
  
  rc = pthread_join(pthr, NULL);

  return 0;
}



