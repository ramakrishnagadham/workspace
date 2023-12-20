#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>

#define N 40000000 // 40 million

int main() {
    int *h_data, *d_data;
    size_t size = N * sizeof(int);

    // Allocate host memory
    h_data = (int*)malloc(size);
    
    // Initialize host array with data
    for (int i = 0; i < N; ++i) {
        h_data[i] = /* your data initialization */;
    }

    // Allocate device memory
    cudaMalloc(&d_data, size);
    
    // Copy data to device
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Allocate temporary storage
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_data, N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Sort data on the device
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_data, N);

    // Copy sorted data back to host
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_data);
    cudaFree(d_temp_storage);
    free(h_data);

    return 0;
}
