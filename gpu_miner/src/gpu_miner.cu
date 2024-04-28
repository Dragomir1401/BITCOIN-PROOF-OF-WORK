#include <stdio.h>
#include <stdint.h>
#include "../include/utils.cuh"
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdarg.h>

void logMessage(const char* format, ...) {
    FILE *fp = fopen("errors.log", "a");
    if (fp != NULL) {
        va_list args;
        va_start(args, format);  // Initialize the argument list with the format
        vfprintf(fp, format, args);  // Print the formatted string to the file
        va_end(args);  // Clean up the argument list
        fprintf(fp, "\n");  // Add a newline after the message
        fclose(fp);
    } else {
        printf("Error opening file!\n");
    }
}

__global__ void findNonce(size_t current_length,
                          BYTE *block_content, 
                          BYTE *block_hash, 
                          size_t max_nonce,
                          size_t nonce_size,
                          uint64_t *d_nonce_result) {
    BYTE difficulty_5_zeros[SHA256_HASH_SIZE] = "00000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = blockDim.x * gridDim.x;

    for (uint64_t nonce = idx; nonce <= max_nonce; nonce += stride) {
        BYTE local_block_content[BLOCK_SIZE];
        memcpy(local_block_content, block_content, current_length);

        BYTE *nonce_position = local_block_content + current_length;

        // Clear the nonce area
        memset(nonce_position, '0', nonce_size);

        // Nonce to ASCII (reversed order)
        uint64_t n = nonce;
        int digit_count = 0;
        while (n > 0) {
            nonce_position[nonce_size - 1 - digit_count++] = '0' + (n % 10);
            n /= 10;
        }

        // Calculate hash
        apply_sha256(local_block_content, current_length + nonce_size, block_hash, 1);

        // Check hash against difficulty
        if (compare_hashes(block_hash, difficulty_5_zeros) == 1) {
            atomicMin((unsigned long long int*)d_nonce_result, (unsigned long long int)nonce);
            break;
        }
    }
}



int main(int argc, char **argv) {
    logMessage("Hello World");

    BYTE hashed_tx1[SHA256_HASH_SIZE], hashed_tx2[SHA256_HASH_SIZE], hashed_tx3[SHA256_HASH_SIZE], hashed_tx4[SHA256_HASH_SIZE],
            tx12[SHA256_HASH_SIZE * 2], tx34[SHA256_HASH_SIZE * 2], hashed_tx12[SHA256_HASH_SIZE], hashed_tx34[SHA256_HASH_SIZE],
            tx1234[SHA256_HASH_SIZE * 2], top_hash[SHA256_HASH_SIZE];

    BYTE *d_block_content, *d_block_hash;
    uint64_t *d_nonce_result, nonce = 0;

    cudaMalloc((void**)&d_block_hash, SHA256_HASH_SIZE);
    cudaMalloc((void**)&d_nonce_result, sizeof(uint64_t));
    cudaMemset(d_nonce_result, 0, sizeof(uint64_t));
    cudaMemset(d_block_hash, 0, SHA256_HASH_SIZE);

    logMessage("Starting GPU miner...");
    logMessage("Allocated memory on device");
    logMessage("Memset done");

	// Top hash
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

	logMessage("Top hash calculated");
	logMessage((const char*)top_hash);

	BYTE block_content[BLOCK_SIZE];
	sprintf((char*)block_content, "%s%s", prev_block_hash, top_hash);
	size_t current_length = strlen((char*) block_content);

	// Copy block content to GPU
    size_t block_size = sizeof(block_content); // Make sure this is defined correctly
    cudaMalloc((void**)&d_block_content, block_size);
    cudaMemcpy(d_block_content, block_content, block_size, cudaMemcpyHostToDevice);

    dim3 blockSize(256); // Example block size
    dim3 gridSize((MAX_NONCE + blockSize.x - 1) / blockSize.x);

    cudaEvent_t start, stop;
    startTiming(&start, &stop);

    logMessage("Starting kernel");
    findNonce<<<gridSize, blockSize>>>(current_length, d_block_content, d_block_hash, MAX_NONCE, NONCE_SIZE, d_nonce_result);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    logMessage("Kernel finished");
    logMessage("Kernel error check done");

    // Copy nonce back to host
    cudaMemcpy(&nonce, d_nonce_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    logMessage("Nonce printed: %" PRIu64, nonce);

    // Copy block hash back to host and print
    BYTE host_block_hash[SHA256_HASH_SIZE];
    cudaMemcpy(host_block_hash, d_block_hash, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
    host_block_hash[SHA256_HASH_SIZE - 1] = '\0'; // Ensure null-termination
    logMessage("Block hash: %s", host_block_hash);

    float seconds = stopTiming(&start, &stop);
    printResult(host_block_hash, nonce, seconds);

    // Free device memory
    cudaFree(d_block_content);
    cudaFree(d_block_hash);
    cudaFree(d_nonce_result);

    return 0;
}