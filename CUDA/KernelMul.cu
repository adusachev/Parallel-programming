#include <iostream>
#include <cmath>

#include "WriteResults.h"




__global__
void KernelMul(int n, float* x, float* y, float* res) {
	/*  
        Поэлементное произведение двух массивов x и y
	*/
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = tid; i < n; i += stride) {
		res[i] = x[i] * y[i];
	}
}


int main(int argc, char *argv[]) {
	int n = 1 << 26;  // 2**28

	// step 1: allocate Host memory
	float *h_x = new float[n];
	float *h_y = new float[n];
	float *h_res = new float[n];

	// step 2: allocate Device memory
	float *d_x;
	float *d_y;
	float *d_res;
	int nbytes = n * sizeof(float);  // size in bytes
	cudaMalloc(&d_x, nbytes);
	cudaMalloc(&d_y, nbytes);
	cudaMalloc(&d_res, nbytes);

	// fill Host arrays
	for (int i = 0; i < n; i++) {
		h_x[i] = 2.0f;
		h_y[i] = 3.0f;
	}

	// step 3: copy arrays from Host to Device (size in bytes!)
	cudaMemcpy(d_x, h_x, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, nbytes, cudaMemcpyHostToDevice);

	// measure calculations time
	cudaEvent_t start, end;
	float milliseconds;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);


	// step 4: run calculations
	int blockSize = atoi(argv[1]);  // get blocksize from command line
	// int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;

	KernelMul<<<numBlocks, blockSize>>>(n, d_x, d_y, d_res);


	// end time measure
	cudaEventRecord(end);

	// step 5: copy calc result from Device to Host
	cudaMemcpy(h_res, d_res, nbytes, cudaMemcpyDeviceToHost);


	cudaEventSynchronize(end);  // (!)

	cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time elapsed: " << milliseconds << " ms " << std::endl;

	// step 6: free Host memory
	delete[] h_x;
	delete[] h_y;
	delete[] h_res;

	// step 7: free Device memory
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_res);

	write_results(blockSize, milliseconds);


	return 0;
}
