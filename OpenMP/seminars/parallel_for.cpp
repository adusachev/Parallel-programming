#include <iostream>
#include <omp.h>


void stupid_parallel_for() {
    /*
    При такой реализации в 1 момент времени суммирование может производить 
     только 1 поток (из-за критической секции).
    То есть теряется смысл в распараллеливании.
    */
    int sum = 0;
    # pragma omp parallel
    {
        # pragma omp for
        for (int i = 0; i < 10000; i++) {
                #pragma omp critical
                {
                    sum += 1;
                }
        }
    }
    
    std::cout << "Result sum: "  << sum << std::endl;
}



void parallel_for_1() {
    /*
    Заводим локальную переменную cur_sum для каждого потока
    Потом вне цикла в критической секции суммруем частисные суммы
    */
    int sum = 0;
    # pragma omp parallel
    {
        int cur_sum = 0;  // локальная переменная
        # pragma omp for
        for (int i = 0; i < 10000; i++) {
                cur_sum += 1;
        }

        #pragma omp critical
        {
            sum += cur_sum;
        }
    }
    
    std::cout << "Result sum: "  << sum << std::endl;
}


void parallel_for_2() {
    /*
    Используем #pragma omp atomic вместо #pragma omp critical  
    */
    int sum = 0;

    # pragma omp parallel
    {
        int cur_sum = 0;
        # pragma omp for
        for (int i = 0; i < 10000; i++) {
                cur_sum += 1;
        }

        # pragma omp atomic
            sum += cur_sum;
    }
    
    std::cout << "Result sum: "  << sum << std::endl;
}


void parallel_for_3() {
    /*
    Sum with reuction:
     внунтри каждого потока создастся локальная копия переменной sum, 
     а на выходе из цикла все локальные суммы просуммируются в общую переменную sum
    */
    int sum = 0;

    # pragma omp parallel
    {
        # pragma omp for reduction(+:sum)
        for (int i = 0; i < 10000; i++) {
                sum += 1;
        }
    }
    std::cout << "Result sum: "  << sum << std::endl;
}


void parallel_for_4() {
    /*
    #pragma omp parallel + #pragma omp for = #pragma omp parallel for 
    */
    int sum = 0;

    # pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < 10000; i++) {
            sum += 1;
    }
    std::cout << "Result sum: "  << sum << std::endl;
}



int main(int argc, char **argv) {
    
    double begin_time = omp_get_wtime(); 


    // stupid_parallel_for();
    // parallel_for_4();


    int sum = 0;
    # pragma omp parallel
    {
        int cur_sum = 0;
        # pragma omp for schedule(dynamic, 1000)
        for (int i = 0; i < 10000; i++) {
                cur_sum += 1;
        }
        # pragma omp atomic
        sum += cur_sum;
    }
    std::cout << "Result sum: "  << sum << std::endl;

    
    double end_time = omp_get_wtime();
    double elapsed_seconds = end_time - begin_time;
    std::cout << "Elapsed time: " << elapsed_seconds << "sec" << std::endl;
}


