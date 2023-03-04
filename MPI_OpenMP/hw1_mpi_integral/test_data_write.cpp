#include<iostream>
#include <fstream>
#include <string>


void write_data(int N, double T_1, double T_p, int n_proc, std::string filename="./results.csv") {
    /*
    Записывает данные в конец файла filename
    */
    std::ofstream out; 
    out.open(filename, std::ios::app);
    out << N << ", " << T_1 << ", " << T_p << ", " << n_proc << "\n";
    out.close();
}


int main() {

    int N = 1000;
    double T_1 = 0.01;
    double T_p = 0.00378;
    int n_proc = 3;

    std::cout << N << std::endl;

    write_data(N, T_1, T_p, n_proc);
    

    return 0;
}



