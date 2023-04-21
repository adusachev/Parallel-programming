#include <iostream>
#include <fstream>
#include <sstream>


void EyeMatrix(float* matrix, int height, int width) {
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if (i == j) {
				matrix[i * width + j] = 1;
			} else {
				matrix[i * width + j] = 0;
			}
		}
	}
}

void OnesMatrix(float* matrix, int height, int width) {
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			matrix[i * width + j] = 1;
		}
	}
}


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

void PrintMatrix(float *matrix, int height, int width) {

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			std::cout << i << " " << j << " " << matrix[i * width + j] << "\n";
		}
	}
}


void MatrixMulCPU(float *A, float *B, float *C, int height, int mid_size, int width) {
    /*
        Matrix multiplication A * B = C
         A(h x m), B(m x w), C(h x w)
    */
    for (int i = 0; i < height; i++) {  // iterate over rows
        for (int j = 0; j < width; j++) {  // iterate over columns
            
            int tmp = 0;
            for (int k = 0; k < mid_size; k++) {
                tmp += A[i * mid_size + k] * B[k * width + j];
            }
            C[i * width + j] = tmp;
        }
    }
}


void save_matrix(float* matrix, int height, int width, std::string filename="./matrix.txt") {
	std::ofstream out; 
	out.open(filename, std::ios::app);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			out << matrix[i * width + j] << " ";
		}
	}
	out << "\n";
}
