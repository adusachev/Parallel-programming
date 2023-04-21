#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "matrix_functions.h"
#include "WriteResults.h"

#define BLOCK_SIZE 16




__global__
void MatrixMul(float* A, float* B, float* C, int mid_size, int width) {
    /* 
        Matrix multiplication A * B = C
    */
    extern __shared__ float shmem_A[BLOCK_SIZE * BLOCK_SIZE];
    extern __shared__ float shmem_B[BLOCK_SIZE * BLOCK_SIZE];

    int col = blockIdx.x * blockDim.x + threadIdx.x;  // column num
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // line num

    float res = .0f;

    for (int k = 0; k < mid_size; k += blockDim.x) {
        shmem_A[threadIdx.y * blockDim.x + threadIdx.x] = A[row * mid_size + k + threadIdx.x];
        shmem_B[threadIdx.y * blockDim.x + threadIdx.x] = B[k * width + col + threadIdx.y * width];
        __syncthreads();

        for (int j = 0; j < blockDim.x; j++) {
            res += shmem_A[threadIdx.y * blockDim.x + j] * shmem_B[j * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }

    C[row * width + col] = res;
}





int main() {

    float *h_A;
    float *h_B;
    float *h_C;
    // int A_height = 128; int A_width = 384;
    // int B_height = 384; int B_width = 256;
    // int C_height = 128; int C_width = 256;
    // int mid_size = 384;

    int A_height = 1280; int A_width = 3840;
    int B_height = 3840; int B_width = 2560;
    int C_height = 1280; int C_width = 2560;
    int mid_size = 3840;

    h_A = new float[A_height * A_width];
    h_B = new float[B_height * B_width];
    h_C = new float[C_height * C_width];

    RandomMatrix(h_A, A_height, A_width, 6);
    RandomMatrix(h_B, B_height, B_width, 6);

    // save_matrix(h_A, A_height, A_width, "A.txt");
    // save_matrix(h_B, B_height, B_width, "B.txt");

    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, 4 * sizeof(float) * A_height * A_width);
    cudaMalloc(&d_B, 4 * sizeof(float) * B_height * B_width);
    cudaMalloc(&d_C, 4 * sizeof(float) * C_height * C_width);

    cudaMemcpy(d_A, h_A, sizeof(float) * A_height * A_width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * B_height * B_width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeof(float) * C_height * C_width, cudaMemcpyHostToDevice);

    // 2D blocks and grid
    int blockSize_x = BLOCK_SIZE;
    int blockSize_y = BLOCK_SIZE;

    int numBlocks_x = (C_width + blockSize_x - 1) / blockSize_x;
    int numBlocks_y = (C_height + blockSize_y - 1) / blockSize_y;

    std::cout << "numBlocks_x = " << numBlocks_x << "; numBlocks_y = " << numBlocks_y << std::endl;
	std::cout << "block_dim.x * num_blocks.x = " << blockSize_x * numBlocks_x 
			  << " >= " << C_width << " = width of C matrix" << std::endl;
	std::cout << "block_dim.y * num_blocks.y = " << blockSize_y * numBlocks_y 
			  << " >= " << C_height << " = height of C matrix" << std::endl << std::endl;

    dim3 block_size(blockSize_x, blockSize_y);
    dim3 num_blocks(numBlocks_x, numBlocks_y);

    // measure calculations time
    cudaEvent_t start, end;
    float milliseconds;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    MatrixMul<<<num_blocks, block_size, (2 * BLOCK_SIZE * BLOCK_SIZE)>>>(d_A, d_B, d_C, mid_size, C_width);

    cudaEventRecord(end);  // end time measure

    cudaMemcpy(h_C, d_C, sizeof(float) * C_height * C_width, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(end);
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time elapsed: " << milliseconds << " ms " << std::endl;

    // save_matrix(h_C, C_height, C_width, "C.txt");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
