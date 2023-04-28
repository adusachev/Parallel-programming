#include <iostream>
#include <cassert>
#include <chrono>



__global__ 
void ScalarMul(int* vec1, int* vec2, int* out_data) {
    /*
        Scalar mul of vectors vec1 and vec2
    */
    extern __shared__ int shmem[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    shmem[tid] = (vec1[index] * vec2[index]) + (vec1[index + blockDim.x] * vec2[index + blockDim.x]);
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shmem[tid] += shmem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_data[blockIdx.x] = shmem[0];
    }
}



int ScalarMulCPU(int n, int* a, int* b) {
    /*
        Scalar mul of vectors a and b with size n
    */
    int res = 0;
    for (int i = 0; i < n; i++) {
        res += (a[i] * b[i]);
    }
    return res;
}




int main() {

    const int block_size = 16;
    const int array_size = 1 << 24;

    // first array
    int* h_vec1 = new int[array_size];
    for (int i = 0; i < array_size; ++i) {
        h_vec1[i] = i;
    }
    // second array
    int* h_vec2 = new int[array_size];
    for (int i = 0; i < array_size; ++i) {
        h_vec2[i] = i + 5;
    }

    int* d_vec1;
    int* d_vec2;
    cudaMalloc(&d_vec1, sizeof(int) * array_size);
    cudaMalloc(&d_vec2, sizeof(int) * array_size);

    cudaMemcpy(d_vec1, h_vec1, sizeof(int) * array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, h_vec2, sizeof(int) * array_size, cudaMemcpyHostToDevice);


    int num_blocks = (array_size + 2 * block_size - 1) / block_size / 2;
    std::cout << "Number of blocks: " << num_blocks << std::endl;

    int* d_blocksum;
    cudaMalloc(&d_blocksum, sizeof(int) * num_blocks);
    int* h_blocksum = new int[num_blocks];

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    ScalarMul<<<num_blocks, block_size, sizeof(int) * block_size>>>(d_vec1, d_vec2, d_blocksum);

    cudaEventRecord(stop);

    cudaMemcpy(h_blocksum, d_blocksum, sizeof(int) * num_blocks, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    
    int res_gpu = 0;
    for (int i = 0; i < num_blocks; ++i) {
        // std::cout << h_blocksum[i] << std::endl; 
        res_gpu += h_blocksum[i];
    }

    std::cout << "Result of scalar mul GPU: " << res_gpu << std::endl;
    std::cout << "GPU time elapsed: " << milliseconds << " ms \n" << std::endl;

    // scalar mul on cpu 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    int res_cpu = ScalarMulCPU(array_size, h_vec1, h_vec2);
    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - begin;
    float milliseconds_cpu = elapsed_seconds.count() * 1000;

    std::cout << "Result of scalar mul CPU = " << res_cpu << std::endl;
    std::cout << "CPU time elapsed = " << milliseconds_cpu << " ms \n" << std::endl;

    std::cout << "Acceleration S = " << milliseconds_cpu / milliseconds << " ms" << std::endl;

    // check that answers form gpu and cpu are equal
    assert(res_cpu == res_gpu);


    cudaFree(d_blocksum);
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    delete[] h_vec1;
    delete[] h_vec2;
    delete[] h_blocksum;
}
