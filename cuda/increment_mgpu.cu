
// increment elements in an array using moderngpu

#include <moderngpu/kernel_transform.hxx>
#include <moderngpu/memory.hxx>
#include <iostream>

const int N = 10;

struct increment_functor {
    __device__ int operator()(int x) {
        return x + 1;
    }
};

int main() {
    mgpu::standard_context_t context;
    mgpu::mem_t<int> d_A(N), d_B(N);
    
    // Initialize input array A
    for (int i = 0; i < N; ++i) {
        d_A[i] = i;
    }
    
    // Perform element-wise increment using ModernGPU
    mgpu::transform(increment_functor(), d_A.get(), d_A.get() + N, d_B.get(), context);
    
    // Copy the result back to host memory
    std::vector<int> h_B(N);
    mgpu::memcpy(&h_B[0], d_B.get(), N * sizeof(int));
    
    // Print the result
    for (int i = 0; i < N; ++i) {
        std::cout << h_B[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
