
#include<iostream>
#include<cassert>
#include <sstream>


int N_from_cli(int argc, char *argv[]) {
    /*
    Returns an integer passed through the command line to the main function
    (e.g.: ./a.out 1000)
    */
    assert (argc == 2);
    int val;
    std::istringstream iss( argv[1] );
    iss >> val;

    // if (argc >= 2) {
    //     std::istringstream iss( argv[1] );
    //     iss >> val;
    // } 
    // else {
    //     std::cout << "There is no arguments in command line" << std::endl;
    //     return 0;
    // }

    return val;
}



int main(int argc, char *argv[]) {


    int N = N_from_cli(argc, argv);

    std::cout << N + 5 << std::endl;



    return 0;
}

