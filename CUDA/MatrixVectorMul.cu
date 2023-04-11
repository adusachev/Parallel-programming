#include <iostream>
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


__global__
void MatrixVectorMul(float* A, float* x, float* y, int width) {
	/* 
		Matrix-vector multiplication A * x = y
	*/
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    y[tid] = .0f;

    for (int k = 0; k < width; k++) {
        y[tid] += A[tid * width + k] * x[k];
    }
}


int main(int argc, char *argv[]) {

	int height = 1280; int width = 3840;
    // int height = 32; int width = 16;

	float *h_A = new float[height * width];  // matrix
    float *h_x = new float[width];  // vector
    float *h_y = new float[height];  // result vector

    // fill array
    for (int i = 0; i < width; i++) {
        h_x[i] = i; 
    }

	// EyeMatrix(h_A, height, width);
	RandomMatrix(h_A, height, width, 10);

	save_matrix(h_A, height, width, "A.txt");

	float* d_A;
	float* d_x;
	float* d_y;
	cudaMalloc(&d_A, sizeof(float) * height * width);
    cudaMalloc(&d_x, sizeof(float) * width);
    cudaMalloc(&d_y, sizeof(float) * height);

    cudaMemcpy(d_A, h_A, sizeof(float) * height * width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, sizeof(float) * width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sizeof(float) * height, cudaMemcpyHostToDevice);

	// int block_size = atoi(argv[1]);  // get blocksize from command line
	int block_size = 32;

	int num_blocks = (height + block_size - 1) / block_size;

	// measure calculations time
	cudaEvent_t start, end;
	float milliseconds;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);


    MatrixVectorMul<<<num_blocks, block_size>>>(d_A, d_x, d_y, width);

	cudaEventRecord(end);  // end time measure

    cudaMemcpy(h_y, d_y, sizeof(float) * height, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&milliseconds, start, end);

    for (int i = 0; i < 4; ++i) {
		std::cout << h_y[i] << "\n";
	}

	std::cout << "Time elapsed: " << milliseconds << " ms " << std::endl;


	cudaFree(d_A);
	cudaFree(d_x);
	cudaFree(d_y);

	delete[] h_A;
	delete[] h_x;
	delete[] h_y;

	// write_results(blockSize_x * blockSize_y, milliseconds);

	return 0;
}
