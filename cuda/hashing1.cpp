// hashing using cudapp - parallel primitives

#include <cudpp/cudpp.h>
#include <iostream>

int main() {
    // Initialize CUDPP
    CUDPPHandle cudppHandle;
    cudppCreate(&cudppHandle);

    // Create the hash table configuration
    CUDPPHashTableConfig config;
    config.type = CUDPP_BASIC_HASH_TABLE;
    config.kvPairsHash = CUDPP_DEFAULT_HASH;
    config.kvPairsEquality = CUDPP_DEFAULT_EQ;
    config.numBuckets = 1024;
    config.maxOccupancy = 0.75;

    // Create the hash table
    CUDPPHandle hashTableHandle;
    cudppHashTableCreate(cudppHandle, &hashTableHandle, &config);

    // Generate some key-value pairs
    int numPairs = 10;
    int* keys = new int[numPairs];
    int* values = new int[numPairs];
    for (int i = 0; i < numPairs; ++i) {
        keys[i] = i;
        values[i] = i * 2;
    }

    // Insert the key-value pairs into the hash table
    cudppHashTableInsert(hashTableHandle, keys, values, numPairs);

    // Perform a lookup
    int keyToLookup = 5;
    int lookupValue;
    cudppHashTableRetrieve(hashTableHandle, &keyToLookup, &lookupValue, 1);

    // Print the lookup result
    std::cout << "Value for key " << keyToLookup << ": " << lookupValue << std::endl;

    // Destroy the hash table
    cudppHashTableDestroy(hashTableHandle);

    // Destroy CUDPP
    cudppDestroy(cudppHandle);

    // Cleanup
    delete[] keys;
    delete[] values;

    return 0;
}
