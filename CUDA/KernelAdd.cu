#include <iostream>
#include <cmath>


__global__ void KernelAdd(int numElements, float* x, float* y, float* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < numElements; i += stride) {
		result[i] = x[i] + y[i];
	}
}



int main() {
	int n = 1 << 28;

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
		h_x[i] = 1.0f;
		h_y[i] = 2.0f;
	}

    // step 3: copy arrays from Host to Device (size in bytes!)
	cudaMemcpy(d_x, h_x, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, nbytes, cudaMemcpyHostToDevice);


    // start measure calculations time
	cudaEvent_t start, end;
	float milliseconds;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

    // step 4: Kernel call
	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;
	KernelAdd<<<numBlocks, blockSize>>>(n, d_x, d_y, d_res);

	// end time measure
	cudaEventRecord(end);

	// step 5: copy calc result from Device to Host
	cudaMemcpy(h_res, d_res, nbytes, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time elapsed: " << milliseconds << " ms " << std::endl;

    // check answer
	float maxError = 0.0f;
	for (int i = 0; i < n; i++) {
		maxError = fmax(maxError, fabs(h_res[i]-3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	cudaFree(d_x);
	cudaFree(d_y);


	return 0;
}