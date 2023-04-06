#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<pthread.h>
#include<math.h>

#include <random>




double uniform_random_double(double left, double right) {

    std::random_device rd;
    std::mt19937 gen(rd());
    // std::default_random_engine rd;  // fix default random seed
 
    std::uniform_real_distribution<double> unif(left, right);
 
    double random_double = unif(rd);
    return random_double;
}

 

double integrate(int N) {
    double x_min = 0;
    double x_max = 3.14;
    double y_min = 0;
    double y_max = 1;
    double z_min = 0;
    double z_max = 3.14 / 2;

    double xi, yi, zi;
    double V_cube = (x_max - x_min) * (y_max - y_min) * (z_max - z_min);

    int n = 0;
    for (int i = 0; i < N; i++) {
        xi = uniform_random_double(x_min, x_min + x_max);
        yi = uniform_random_double(y_min, y_min + y_max);
        zi = uniform_random_double(z_min, z_min + z_max);

        if ((yi <= sin(xi)) && (zi <= xi * yi)) {
            n += 1;
        }
    }

    double ans_monte_carlo = ((double)n / N) * V_cube;
    
    return ans_monte_carlo;
}



int main() {

    // std::cout << uniform_random_double(0, 1) << std::endl; 

    std::cout << integrate(100000) << std::endl;














    // std::random_device rd;  // Will be used to obtain a seed for the random number engine
    // std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

    // // std::default_random_engine rd;  // fix default random seed

    // // Declaring the upper and lowerbounds
    // double lower_bound = 0;
    // double upper_bound = 100;
 
    // std::uniform_real_distribution<double> unif(lower_bound,
    //                                             upper_bound);
 
    // // Getting a random double value
    // double random_double = unif(rd);
 
    // std::cout << random_double << std::endl;




    return 0;
}




