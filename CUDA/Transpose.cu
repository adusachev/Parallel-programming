#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>

#include "matrix_functions.h"
#include "WriteResults.h"



void TransposeMatrix(float *matrix, int height, int width, float *result) {
    /*
        Transpose flatten matrix on cpu
        Transposed matrix will be written to result
    */
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            result[j * height + i] = matrix[i * width + j];
        }
    }
}



__global__
void Transpose(float* B, float* BT, int height, int width) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    BT[j * height + i] = B[i * width + j];
}



int main() {

	float *h_B;  // matrix
    float *h_BT;  // transposed B on gpu
    float *cpu_BT;  // transposed B on cpu

	int B_height = 3840; int B_width = 2560;

	// int B_height = 384; int B_width = 256;


	h_B = new float[B_height * B_width];
    h_BT = new float[B_height * B_width];
    cpu_BT = new float[B_height * B_width];



	RandomMatrix(h_B, B_height, B_width, 6);

    // transpose matrix on cpu and measure time
    auto start_cpu = std::chrono::steady_clock::now();
    
    TransposeMatrix(h_B, B_height, B_width, cpu_BT);

    auto end_cpu = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_cpu - start_cpu;
    std::cout << "Time elapsed on cpu: " << elapsed_seconds.count() * 1000 << " ms \n";


	float* d_B;
	float* d_BT;
	cudaMalloc(&d_B, sizeof(float) * B_height * B_width);
	cudaMalloc(&d_BT, sizeof(float) * B_height * B_width);

    cudaMemcpy(d_B, h_B, sizeof(float) * B_height * B_width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_BT, h_BT, sizeof(float) * B_height * B_width, cudaMemcpyHostToDevice);

	// 2D blocks and grid
	// int BS = atoi(argv[1]);  // get blocksize from command line
	int BS = 16;
	int blockSize_x = BS;
	int blockSize_y = BS;


	int numBlocks_x = (B_height + blockSize_x - 1) / blockSize_x;
	int numBlocks_y = (B_width + blockSize_y - 1) / blockSize_y;

	// std::cout << "numBlocks_x = " << numBlocks_x << "; numBlocks_y = " << numBlocks_y << std::endl;
	// std::cout << "block_dim.x * num_blocks.x = " << blockSize_x * numBlocks_x 
	// 		  << " <= " << BT_height << " = height of C matrix" << std::endl;
	// std::cout << "block_dim.y * num_blocks.y = " << blockSize_y * numBlocks_y 
	// 		  << " <= " << BT_width << " = width of C matrix" << std::endl;

	dim3 block_size(blockSize_x, blockSize_y);
	dim3 num_blocks(numBlocks_x, numBlocks_y);


	// measure calculations time
	cudaEvent_t start, end;
	float milliseconds;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

    Transpose<<<num_blocks, block_size>>>(d_B, d_BT, B_height, B_width);

	cudaEventRecord(end);  // end time measure

    cudaMemcpy(h_BT, d_BT, sizeof(float) * B_height * B_width, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&milliseconds, start, end);
	std::cout << "Time elapsed on gpu: " << milliseconds << " ms " << std::endl;

    // compare answers form cpu and gpu
    int counter = 0;
    for (int i = 0; i < B_height * B_width; i++) {
        if (cpu_BT[i] != h_BT[i]) {
            counter++;
            std::cout << "Fail transpose at index " << i << std::endl;
        }
    }
    if (counter == 0) {
        std::cout << "Transposed succesful" << std::endl;
    }

	cudaFree(d_B);
	cudaFree(d_BT);

	delete[] h_B;
	delete[] h_BT;

	return 0;
}
