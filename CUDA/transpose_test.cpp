#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


void RandomMatrix(float* matrix, int height, int width, int max_num) {
	/*
		Fill matrix with random ints from 0 to (max_num-1)
	*/
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			matrix[i * width + j] = rand() % max_num;
		}
	}
}


void TransposeMatrix(float *matrix, int height, int width, float *result) {
    /*
        Transpose flatten matrix
        Transposed matrix will be written to result
    */
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            result[j * height + i] = matrix[i * width + j];
        }
    }
}



void save_matrix(float* matrix, int height, int width, std::string filename="./generated_matrix.csv") {
	std::ofstream out; 
	out.open(filename, std::ios::app);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			out << matrix[i * width + j] << " ";
		}
	}
	out << "\n";
}




void transpose_test() {

    // int A_height = 128; int A_width = 384;
	// int AT_height = 384; int AT_width = 128;
    int A_height = 3; int A_width = 4;
	int AT_height = 4; int AT_width = 3;

    float *A; float *AT;
	A = new float[A_height * A_width];
	AT = new float[AT_height * AT_width];

	RandomMatrix(A, A_height, A_width, 6);
	save_matrix(A, A_height, A_width, "A.txt");

    TransposeMatrix(A, A_height, A_width, AT);
	save_matrix(AT, AT_height, AT_width, "B.txt");

    delete[] A;
	delete[] AT;
}





 
int main() {

    transpose_test();


    // int N = 2;
    // // This program will create some sequence of random numbers on every program run within range 0 to N-1
    // for (int i = 0; i < 20; i++) {
    //     std::cout << rand() % N << " ";
    // }

    return 0;
}

