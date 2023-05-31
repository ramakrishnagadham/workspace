// List ranking using moderngpu

#include <moderngpu/kernel_merge.hxx>
#include <moderngpu/memory.hxx>
#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_scan.hxx>

int main() {
  int count = 10;

  // Allocate GPU memory for input and output arrays
  mgpu::standard_context_t context;
  mgpu::device_memory<int> successor(count, context);
  mgpu::device_memory<int> ranks(count, context);
  mgpu::device_memory<int> hasChanged(1, context);

  // Initialize successor array on the host
  std::vector<int> hostSuccessor = {2, 0, 4, 3, 1, 6, 7, 5, 9, 8};

  // Transfer successor array from host to device
  mgpu::memcpy(successor.data(), hostSuccessor.data(), count);

  int numRounds = static_cast<int>(std::ceil(std::log2(count)));

  // Compute the ranks using pointer jumping algorithm with log n rounds
  for (int round = 0; round < numRounds; ++round) {
    hasChanged[0] = 0;

    mgpu::transform([=] MGPU_DEVICE(int index) {
      int rank = successor[index];
      successor[index] = (rank != -1) ? successor[rank] : -1;
      if (successor[index] != rank)
        hasChanged[0] = 1;
      return rank;
    }, ranks.data(), count, context);

    mgpu::synchronize(context);

    if (!hasChanged[0])
      break;
  }

  // Transfer ranks array from device to host
  std::vector<int> hostRanks(count);
  mgpu::memcpy(hostRanks.data(), ranks.data(), count, context);

  // Print the ranks
  for (int i = 0; i < count; ++i) {
    std::cout << hostRanks[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
