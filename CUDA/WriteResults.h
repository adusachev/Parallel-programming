#include <fstream>
#include <string>
#include <sstream>



void write_results(int block_size, double time, std::string filename="./results.csv") {
    /*
    	Write block_size and time values to the end of the file "filename"
    */
    std::ofstream out; 
    out.open(filename, std::ios::app);
    out << block_size << ", " << time << "\n";
    out.close();
}

