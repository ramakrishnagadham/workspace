#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <iostream>

int main() {
    int N = 10;
    thrust::device_vector<int> input(N);

    // Initialize or populate the input vector with duplicates
    input[0] = 2;
    input[1] = 4;
    input[2] = 1;
    input[3] = 2;
    input[4] = 4;
    input[5] = 6;
    input[6] = 3;
    input[7] = 5;
    input[8] = 3;
    input[9] = 5;

    // Sort the input vector
    thrust::sort(input.begin(), input.end());

    // Remove duplicates
    auto new_end = thrust::unique(input.begin(), input.end());
    input.erase(new_end, input.end());

    // Print the result
    std::cout << "Array without duplicates: ";
    for (int i = 0; i < input.size(); ++i) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
