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
void MatrixMul(float* A, float* B, float* C, int mid_size) {
	/* 
		Matrix multiplication A * B = C
	*/
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // line num
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // column num

    int height = blockDim.x * gridDim.x;
    int width = blockDim.y * gridDim.y;

    C[i * width + j] = .0f;

    for (int k = 0; k < mid_size; ++k) {
        C[i * width + j] += A[i * mid_size + k] * B[k * width + j];
    }
}


int main(int argc, char *argv[]) {
	float *h_A;
	float *h_B;
	float *h_C;
	int A_height = 128; int A_width = 384;
	int B_height = 384; int B_width = 256;
	int C_height = 128; int C_width = 256;
	int mid_size = 384;

	// int A_height = 5; int A_width = 6;
	// int B_height = 6; int B_width = 7;
	// int C_height = 5; int C_width = 7;
	// int mid_size = 6;

	h_A = new float[A_height * A_width];
	h_B = new float[B_height * B_width];
	h_C = new float[C_height * C_width];

	// EyeMatrix(h_A, A_height, A_width);
	// EyeMatrix(h_B, B_height, B_width);
	RandomMatrix(h_A, A_height, A_width, 12);
	RandomMatrix(h_B, B_height, B_width, 12);

	save_matrix(h_A, A_height, A_width, "A.txt");
	save_matrix(h_B, B_height, B_width, "B.txt");

    // PrintMatrix(h_A, A_height, A_width);

	float* d_A;
	float* d_B;
	float* d_C;
	cudaMalloc(&d_A, sizeof(float) * A_height * A_width);
	cudaMalloc(&d_B, sizeof(float) * B_height * B_width);
	cudaMalloc(&d_C, sizeof(float) * C_height * C_width);

    cudaMemcpy(d_A, h_A, sizeof(float) * A_height * A_width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * B_height * B_width, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, sizeof(float) * C_height * C_width, cudaMemcpyHostToDevice);

	// 2D blocks and grid
	int BS = atoi(argv[1]);  // get blocksize from command line
	// int BS = 16;
	int blockSize_x = BS;
	int blockSize_y = BS;

	// want:  block_dim.x * num_blocks.x = height_C - size od matrix C
    //        block_dim.y * num_blocks.y = width_C
	int numBlocks_x = (C_height + blockSize_x - 1) / blockSize_x;
	int numBlocks_y = (C_width + blockSize_y - 1) / blockSize_y;

	std::cout << "numBlocks_x = " << numBlocks_x << "; numBlocks_y = " << numBlocks_y << std::endl;
	std::cout << "block_dim.x * num_blocks.x = " << blockSize_x * numBlocks_x 
			  << " <= " << C_height << " = height of C matrix" << std::endl;
	std::cout << "block_dim.y * num_blocks.y = " << blockSize_y * numBlocks_y 
			  << " <= " << C_width << " = width of C matrix" << std::endl;

	dim3 block_size(blockSize_x, blockSize_y);
	dim3 num_blocks(numBlocks_x, numBlocks_y);

	// measure calculations time
	cudaEvent_t start, end;
	float milliseconds;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);


    MatrixMul<<<num_blocks, block_size>>>(d_A, d_B, d_C, mid_size);

	cudaEventRecord(end);  // end time measure

    cudaMemcpy(h_C, d_C, sizeof(float) * C_height * C_width, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&milliseconds, start, end);
	std::cout << "Time elapsed: " << milliseconds << " ms " << std::endl;

	save_matrix(h_C, C_height, C_width, "C.txt");

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	write_results(blockSize_x, milliseconds);

	return 0;
}
