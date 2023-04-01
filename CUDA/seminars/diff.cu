
#include <iostream>
#include <cmath>
#include <cstdio>



// __global__
// void DiffKernelGlobal(int n, int* u, int *delta) {
// 	int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid + 1 < n) {
//         delta[tid] = u[tid+1] - u[tid];
//     }    
// }


// __global__
// void DiffKernel(int n, int* u, int *delta) {
//     extern __shared__ int shmem[];  //
// 	int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     shmem[threadIdx.x] = u[tid];

//     __syncthreads();

//     if (threadIdx.x + 1 < blockDim.x) {
//         delta[tid] = shmem[threadIdx.x + 1] - shmem[threadIdx.x];
//     } 
//     else if (tid + 1 < n) {
//         delta[tid] = u[tid + 1] - shmem[threadIdx.x];
//     }
// }



__global__
void DiffKernelGlobal(int n, int* u, int *delta) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid + 1 < n) {
        for (int i = 0; i < 100; i++) {
            delta[tid] += u[tid+1] - u[tid];
        }
    }    
}


__global__
void DiffKernel(int n, int* u, int *delta) {
    extern __shared__ int shmem[];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

    shmem[threadIdx.x] = u[tid];

    __syncthreads();

    int result = 0;
    for (int i = 0; i < 100; i++) {
        // if (threadIdx.x + 1 < blockDim.x) {
        // delta[tid] = shmem[threadIdx.x + 1] - shmem[threadIdx.x];
        // } 
        // else if (tid + 1 < n) {
        //     delta[tid] += u[tid + 1] - shmem[threadIdx.x];
        // }

        // one way to optimize
        if (threadIdx.x + 1 < blockDim.x) {
        result += shmem[threadIdx.x + 1] - shmem[threadIdx.x];
        }
        else if (tid + 1 < n) {
            result += u[tid + 1] - shmem[threadIdx.x];
        }
    }

    delta[tid] = result;
}



int main() {

    int* h_u;
    int* d_u;

    int* h_delta;
    int* d_delta;

    // int n = 512 * 8;
    int n = 1 << 24;
    h_u = new int[n];
    h_delta = new int[n];

    unsigned int size = n * sizeof(int);
    cudaMalloc(&d_u, size);
	cudaMalloc(&d_delta, size);

    for (int i = 0; i < n; i++) {
        h_u[i] = i;
    }

    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_delta, h_delta, size, cudaMemcpyHostToDevice);

    // start measure calculations time
	cudaEvent_t start, end;
	float milliseconds;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

    // DiffKernelGlobal<<<16, 256>>>(n, d_u, d_delta);
    // DiffKernel<<<16, 256, 256 * sizeof(int)>>>(n, d_u, d_delta);  // 256 * sizeof(int) - размер shared memeory
    
    // DiffKernelGlobal<<<1 << 16, 256>>>(n, d_u, d_delta);
    DiffKernel<<<1 << 16, 256, 256 * sizeof(int)>>>(n, d_u, d_delta);  // 256 * sizeof(int) - размер shared memeory

    // end time measure
	cudaEventRecord(end);

    cudaMemcpy(h_delta, d_delta, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(end);
	cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Time elapsed: " << milliseconds << " ms " << std::endl;

    // for (int i = 0; i < n; i++) {
    //     std::cout << h_delta[i] << " ";
    // }
    // std::cout << std::endl;

    cudaFree(d_u);
	cudaFree(d_delta);
	delete[] h_u;
	delete[] h_delta;




	return 0;
}




















































