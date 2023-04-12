#include <iostream>

#define BLOCK_SIZE 256


#include "matrix_functions.h"
#include "WriteResults.h"







__global__ void KernelMatrixAdd(int height, int width, float* A, float* B, float* result) {
    /* 
		Matrix sum: A + B = C
	*/
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // line num
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // column num 
    
    result[i * width + j] = A[i * width + j] + B[i * width + j];
    
}



int main(int argc, char *argv[]) {
	float *h_A;
	float *h_B;
	float *h_C;

    int height = 1280;
    int width = 2560;
    // int height = 3;
    // int width = 3;

	h_A = new float[height * width];  // выделяем матрицы как flatten массивы
	h_B = new float[height * width];
	h_C = new float[height * width];

	EyeMatrix(h_A, height, width);
	EyeMatrix(h_B, height, width);
	// OnesMatrix(h_A, height, width);
	// OnesMatrix(h_B, height, width);

    // PrintMatrix(h_A, height, width);

	float* d_A;
	float* d_B;
	float* d_C;
	cudaMalloc(&d_A, sizeof(float) * height * width);
	cudaMalloc(&d_B, sizeof(float) * height * width);
	cudaMalloc(&d_C, sizeof(float) * height * width);

    cudaMemcpy(d_A, h_A, sizeof(float) * height * width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * height * width, cudaMemcpyHostToDevice);

    // measure calculations time
	cudaEvent_t start, end;
	float milliseconds;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

    // 2D blocks and grid
	int BS = atoi(argv[1]);  // get blocksize from command line
	// int BS = 16;
	int blockSize_x = BS;
	int blockSize_y = BS;
    dim3 block_size(blockSize_x, blockSize_y);  // each block has 16x16 threads

	// want:  block_dim.x * num_blocks.x = height
    //        block_dim.y * num_blocks.y = width
	int numBlocks_x = (height + blockSize_y - 1) / blockSize_x;
	int numBlocks_y = (width + blockSize_x - 1) / blockSize_x;
	dim3 num_blocks(numBlocks_x, numBlocks_y);

	// dim3 num_blocks(8, 16);  // 8x16 blocks

    std::cout << "numBlocks_x = " << numBlocks_x << "; numBlocks_y = " << numBlocks_y << std::endl;
	std::cout << "block_dim.x * num_blocks.x = " << blockSize_x * numBlocks_x << " <= " << height << " = height" << std::endl;
	std::cout << "block_dim.y * num_blocks.y = " << blockSize_y * numBlocks_y << " <= " << width << " = width" << std::endl;


    KernelMatrixAdd<<<num_blocks, block_size>>>(height, width, d_A, d_B, d_C);

    cudaEventRecord(end);
    cudaMemcpy(h_C, d_C, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(end);
	cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time elapsed: " << milliseconds << " ms " << std::endl;

    PrintMatrix(h_C, height, width);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	write_results(blockSize_x, milliseconds);

	return 0;
}
