#include <stdio.h>
#include <stdint.h>
#include "../include/utils.cuh"
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdarg.h>

__device__ int atomic_found = 0;
__device__ BYTE global_difficulty_5_zeros[SHA256_HASH_SIZE];

__global__ void findNonce(BYTE *block_content, 
                            BYTE *block_hash, 
                            uint64_t *d_nonce_result,
                            size_t current_length)
{

    // If the atomic_found is set to 1, return
    if (atomic_found == 1) {
        return;
    }

    // Get the thread index
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert index to string
    char nonce_string[NONCE_SIZE + 1];
    intToString(idx, nonce_string);

    // Copy block content and nonce to local memory
    BYTE local_block_content[SHA256_BLOCK_SIZE + NONCE_SIZE];
    d_strcpy((char*)local_block_content, (char*)block_content);
    d_strcpy((char*)local_block_content + current_length, nonce_string);

    // Check again the atomic_found to ensure the time consuming sha256 is not executed
    // if the hash is already found
    if (atomic_found == 1) {
        return;
    }

    // Calculate hash
    BYTE local_block_hash[SHA256_HASH_SIZE]; 
    apply_sha256(local_block_content, d_strlen((char*)local_block_content), local_block_hash, 1);

    // If the atomic_found is set to 1, don't check the hash with this thread because 
    // another thread has already found the hash
    if (atomic_found == 1) {
        return;
    }

    // Check hash against difficulty
    if (compare_hashes(local_block_hash, global_difficulty_5_zeros) <= 0) {
        // Try setting the atomic_found to 1, ensuring only one thread can proceed
        if (atomicCAS(&atomic_found, 0, 1) == 0) {
            printf("Hash found: %s\n", local_block_hash);

            // Double-check if another thread has not already written the result
            if (*d_nonce_result == 0) {
                // Write the nonce result
                *d_nonce_result = idx;

                // Write the block hash
                d_strcpy((char*)block_hash, (char*)local_block_hash);
            }
        }
    }
}


int main(int argc, char **argv) {
    // Copy to file global difficulty 5 zeros template
    cudaMemcpyToSymbol(global_difficulty_5_zeros, DIFFICULTY, SHA256_HASH_SIZE);

    BYTE hashed_tx1[SHA256_HASH_SIZE],
        hashed_tx2[SHA256_HASH_SIZE],
        hashed_tx3[SHA256_HASH_SIZE],
        hashed_tx4[SHA256_HASH_SIZE],
        tx12[SHA256_HASH_SIZE * 2],
        tx34[SHA256_HASH_SIZE * 2],
        hashed_tx12[SHA256_HASH_SIZE],
        hashed_tx34[SHA256_HASH_SIZE],
        tx1234[SHA256_HASH_SIZE * 2],
        top_hash[SHA256_HASH_SIZE];

    // Declare device memory pointers
    BYTE *d_block_content, *d_block_hash;
    uint64_t *d_nonce_result, nonce = 0;

    // Allocate memory on device
    cudaMalloc((void**)&d_block_hash, SHA256_HASH_SIZE);
    cudaMalloc((void**)&d_nonce_result, sizeof(uint64_t));

    // Initialize device memory
    cudaMemset(d_nonce_result, 0, sizeof(uint64_t));
    cudaMemset(d_block_hash, 0, SHA256_HASH_SIZE);

	// Calculate top hash
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

    // Calculate block content
	BYTE block_content[BLOCK_SIZE];
	sprintf((char*)block_content, "%s%s", prev_block_hash, top_hash);
	size_t current_length = strlen((char*) block_content);

	// Copy block content to GPU
    cudaMalloc((void**)&d_block_content, BLOCK_SIZE);
    cudaMemcpy(d_block_content, block_content, BLOCK_SIZE, cudaMemcpyHostToDevice);
    printf("Block content: %s\n", block_content);

    // Declare and initialize grid and block sizes
    dim3 blockSize(256);
    dim3 gridSize((MAX_NONCE + blockSize.x - 1) / blockSize.x);

    // Start timing
    cudaEvent_t start, stop;
    startTiming(&start, &stop);

    // Call kernel
    findNonce<<<gridSize, blockSize>>>(d_block_content, d_block_hash, d_nonce_result, current_length);
    cudaDeviceSynchronize();
    
    // Stop timing
    float seconds = stopTiming(&start, &stop);

    // Copy nonce back to host
    cudaMemcpy(&nonce, d_nonce_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Copy block hash back to host and print
    BYTE host_block_hash[SHA256_HASH_SIZE];
    cudaMemcpy(host_block_hash, d_block_hash, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
    host_block_hash[SHA256_HASH_SIZE - 1] = '\0';

    // Print result
    printResult(host_block_hash, nonce, seconds);

    // Free device memory
    cudaFree(d_block_content);
    cudaFree(d_block_hash);
    cudaFree(d_nonce_result);

    return 0;
}