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
