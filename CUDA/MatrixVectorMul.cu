#include <iostream>

#include "matrix_functions.h"
#include "WriteResults.h"



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

	int height = 1024; int width = 3840;
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
	// save_matrix(h_A, height, width, "A.txt");


	float* d_A;
	float* d_x;
	float* d_y;
	cudaMalloc(&d_A, sizeof(float) * height * width);
    cudaMalloc(&d_x, sizeof(float) * width);
    cudaMalloc(&d_y, sizeof(float) * height);

    cudaMemcpy(d_A, h_A, sizeof(float) * height * width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, sizeof(float) * width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sizeof(float) * height, cudaMemcpyHostToDevice);

	int blockSize = atoi(argv[1]);  // get blocksize from command line
	// int blockSize = 32;

	int num_blocks = (height + blockSize - 1) / blockSize;
	std::cout << "blockSize = " << blockSize << "; num_blocks = " << num_blocks << std::endl;

	// measure calculations time
	cudaEvent_t start, end;
	float milliseconds;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);


    MatrixVectorMul<<<num_blocks, blockSize>>>(d_A, d_x, d_y, width);

	cudaEventRecord(end);  // end time measure

    cudaMemcpy(h_y, d_y, sizeof(float) * height, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&milliseconds, start, end);


	std::cout << "Time elapsed: " << milliseconds << " ms " << std::endl;


	cudaFree(d_A);
	cudaFree(d_x);
	cudaFree(d_y);

	delete[] h_A;
	delete[] h_x;
	delete[] h_y;

	write_results(blockSize, milliseconds);

	return 0;
}
