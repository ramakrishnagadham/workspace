
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Structure to hold student details
struct StudentDetails {
    int numStudents;
    int* rollNumbers;
    int* mathsMarks;
    int* physicsMarks;
    int* totalMarks;
};

// CUDA kernel to compute total marks for each student
__global__ void computeTotalMarks(StudentDetails details) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < details.numStudents) {
        details.totalMarks[tid] = details.mathsMarks[tid] + details.physicsMarks[tid];
    }
}

int main() {
    const int numStudents = 1000;

    // Allocate host vectors
    std::vector<int> hostRollNumbers(numStudents);
    std::vector<int> hostMathsMarks(numStudents);
    std::vector<int> hostPhysicsMarks(numStudents);
    std::vector<int> hostTotalMarks(numStudents);

    // Generate random student details
    for (int i = 0; i < numStudents; ++i) {
        hostRollNumbers[i] = i + 1;  // Assuming roll numbers start from 1
        hostMathsMarks[i] = rand() % 101;  // Random marks between 0 and 100
        hostPhysicsMarks[i] = rand() % 101;
    }

    // Allocate device vectors
    StudentDetails deviceDetails;
    deviceDetails.numStudents = numStudents;
    cudaMalloc((void**)&deviceDetails.rollNumbers, numStudents * sizeof(int));
    cudaMalloc((void**)&deviceDetails.mathsMarks, numStudents * sizeof(int));
    cudaMalloc((void**)&deviceDetails.physicsMarks, numStudents * sizeof(int));
    cudaMalloc((void**)&deviceDetails.totalMarks, numStudents * sizeof(int));

    // Copy host vectors to device vectors
    cudaMemcpy(deviceDetails.rollNumbers, hostRollNumbers.data(), numStudents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDetails.mathsMarks, hostMathsMarks.data(), numStudents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDetails.physicsMarks, hostPhysicsMarks.data(), numStudents * sizeof(int), cudaMemcpyHostToDevice);

    // Invoke the CUDA kernel
    int blockSize = 256;
    int gridSize = (numStudents + blockSize - 1) / blockSize;
    computeTotalMarks<<<gridSize, blockSize>>>(deviceDetails);

    // Copy device vector back to host
    cudaMemcpy(hostTotalMarks.data(), deviceDetails.totalMarks, numStudents * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < numStudents; ++i) {
        std::cout << "Roll Number: " << hostRollNumbers[i] << ", Total Marks: " << hostTotalMarks[i] << std::endl;
    }

    // Free device memory
    cudaFree(deviceDetails.rollNumbers);
    cudaFree(deviceDetails.mathsMarks);
    cudaFree(deviceDetails.physicsMarks);
    cudaFree(deviceDetails.totalMarks);

    return 0;
}
