#include <stdio.h>
#include <stdint.h>
#include "../include/utils.cuh"
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdarg.h>
#define THREAD_BLOCK_SIZE 256

__device__ bool found = 0;

__global__ void findNonce(BYTE *block_hash_param, BYTE *block_content_param, uint64_t *nonce_device, size_t current_size)
{
    // If another thread has already found the hash, return
    if (found == true) {
        return;
    }

    // Declare the hash to test against
    BYTE difficulty_5_zeros[SHA256_HASH_SIZE] = "0000099999999999999999999999999999999999999999999999999999999999";
    
    // Compute the thread index
    uint64_t idx = blockIdx.x;
    idx *= blockDim.x;
    idx += threadIdx.x;

    // Convert the index to string
    char string[NONCE_SIZE];
    intToString(idx, string);
    
    // Copy the block content and nonce to local memory to ensure thread safety on these variables
    BYTE content_local_thread[SHA256_BLOCK_SIZE + NONCE_SIZE];
    BYTE hash_local_thread[SHA256_HASH_SIZE]; 
    d_strcpy((char*)content_local_thread, (char*)block_content_param);
    d_strcpy((char*)content_local_thread + current_size, string);

    // Calculate the hash
    apply_sha256(content_local_thread, d_strlen((char*)content_local_thread), hash_local_thread, 1);

    // Check if the hash is below the difficulty
    if (compare_hashes(hash_local_thread, difficulty_5_zeros) <= 0) {
        // Write the result to the parameters variables and set flag to true
        *nonce_device = idx;
        d_strcpy((char*)block_hash_param, (char*)hash_local_thread);
        found = true;
    }
}


int main(int argc, char **argv) {
    // Define the buffers to apply hashing
    BYTE hashed_tx1[SHA256_HASH_SIZE];
    BYTE hashed_tx2[SHA256_HASH_SIZE];
    BYTE hashed_tx3[SHA256_HASH_SIZE];
    BYTE hashed_tx4[SHA256_HASH_SIZE];
    BYTE tx12[SHA256_HASH_SIZE * 2];
    BYTE tx34[SHA256_HASH_SIZE * 2];
    BYTE hashed_tx12[SHA256_HASH_SIZE];
    BYTE hashed_tx34[SHA256_HASH_SIZE];
    BYTE tx1234[SHA256_HASH_SIZE * 2];
    BYTE top_hash[SHA256_HASH_SIZE];

    // Apply sha256
	apply_sha256(tx1, strlen((const char*)tx1), hashed_tx1, 1);
	apply_sha256(tx2, strlen((const char*)tx2), hashed_tx2, 1);
	apply_sha256(tx3, strlen((const char*)tx3), hashed_tx3, 1);
	apply_sha256(tx4, strlen((const char*)tx4), hashed_tx4, 1);
	strcpy((char *)tx12, (const char *)hashed_tx1);
	strcat((char *)tx12, (const char *)hashed_tx2);
	apply_sha256(tx12, strlen((const char*)tx12), hashed_tx12, 1);
	strcpy((char *)tx34, (const char *)hashed_tx3);
	strcat((char *)tx34, (const char *)hashed_tx4);
	apply_sha256(tx34, strlen((const char*)tx34), hashed_tx34, 1);
	strcpy((char *)tx1234, (const char *)hashed_tx12);
	strcat((char *)tx1234, (const char *)hashed_tx34);
	apply_sha256(tx1234, strlen((const char*)tx34), top_hash, 1);

    // Declare all needed variable
    cudaEvent_t start, stop;
    BYTE *block_content_device;
    BYTE *block_hash_device;
    BYTE host_block_hash[SHA256_HASH_SIZE];
	BYTE block_content[BLOCK_SIZE];
    uint64_t *nonce_device = nullptr;
    uint64_t nonce = 0;
    float seconds = 0;
    uint64_t current_size = 0;

    // Alloc memory on device for all variables
    cudaMalloc((void**)&block_content_device, BLOCK_SIZE);
    cudaMalloc((void**)&block_hash_device, SHA256_HASH_SIZE);
    cudaMalloc((void**)&nonce_device, sizeof(uint64_t));

    // Set memory on device
    cudaMemset(nonce_device, 0, sizeof(uint64_t));
	sprintf((char*)block_content, "%s%s", prev_block_hash, top_hash);
	current_size = strlen((char*) block_content);
    cudaMemcpy(block_content_device, block_content, BLOCK_SIZE, cudaMemcpyHostToDevice);

    // Define the grid and block size
    dim3 block(THREAD_BLOCK_SIZE);
    uint64_t x = block.x;
    dim3 grid((MAX_NONCE + x - 1) / x);

    // Start timing
    startTiming(&start, &stop);
    
    // Call the kernel
    findNonce<<<grid, block>>>(block_hash_device, block_content_device, nonce_device, current_size);

    // Sync and get the time
    cudaDeviceSynchronize();
    seconds = stopTiming(&start, &stop);

    // Write the results back to host
    cudaMemcpy(&nonce, nonce_device, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_block_hash, block_hash_device, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);

    // Print the results
    printResult(host_block_hash, nonce, seconds);

    // Free the memory
    cudaFree(block_content_device);
    cudaFree(block_hash_device);
    cudaFree(nonce_device);

    return 0;
}