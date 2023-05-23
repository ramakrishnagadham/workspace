#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <iostream>

struct customFunctor {
    __host__ __device__
    int operator()(int x) const {
        // Perform some transformation
        return x * 2;
    }
};

int main() {
    int N = 10;
    thrust::device_vector<int> input(N);

    // Initialize or populate the input vector
    for (int i = 0; i < N; ++i) {
        input[i] = i + 1;
    }

    // Combined operation: transform and reduce
    int result = thrust::transform_reduce(
        input.begin(), input.end(), customFunctor(), 0, thrust::plus<int>()
    );

    // Print the result
    std::cout << "Result: " << result << std::endl;

    return 0;
}
