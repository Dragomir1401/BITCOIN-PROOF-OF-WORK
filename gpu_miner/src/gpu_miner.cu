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

__global__ void findNonce(BYTE *block_content, 
                          BYTE *block_hash, 
                          uint64_t *d_nonce_result)
{
    BYTE difficulty_5_zeros[SHA256_HASH_SIZE] = "0000099999999999999999999999999999999999999999999999999999999999";
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    char nonce_string[NONCE_SIZE + 1];
    intToString(idx, nonce_string);

    BYTE local_block_content[BLOCK_SIZE + NONCE_SIZE];
    d_strcpy((char*)local_block_content, (char*)block_content);
    d_strcpy((char*)local_block_content + BLOCK_SIZE, nonce_string);

    BYTE local_block_hash[SHA256_HASH_SIZE]; 

    // Calculate hash
    apply_sha256(local_block_content, BLOCK_SIZE + NONCE_SIZE, local_block_hash, 1);

    // Check hash against difficulty
    if (compare_hashes(local_block_hash, difficulty_5_zeros) <= 0) {
        printf("Hash found: %s\n", local_block_hash);
        *d_nonce_result = idx;
        d_strcpy((char*)block_hash, (char*)local_block_hash);
        return;
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
    size_t block_size = sizeof(block_content);
    cudaMalloc((void**)&d_block_content, block_size);
    cudaMemcpy(d_block_content, block_content, block_size, cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((MAX_NONCE + blockSize.x - 1) / blockSize.x);

    cudaEvent_t start, stop;
    startTiming(&start, &stop);
    logMessage("Starting kernel");
    logMessage("Block size: %zu", block_size);
    logMessage("Grid size: %d", gridSize.x);
    logMessage("Block size: %d", blockSize.x);
    logMessage("Current length: %zu", current_length);
    logMessage("Block content: %s", block_content);

    findNonce<<<gridSize, blockSize>>>(d_block_content, d_block_hash, d_nonce_result);
    
    logMessage("Kernel finished");
    float seconds = stopTiming(&start, &stop);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    logMessage("Kernel error check done");

    // Copy nonce back to host
    cudaMemcpy(&nonce, d_nonce_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    logMessage("Nonce printed: %" PRIu64, nonce);

    // Copy block hash back to host and print
    BYTE host_block_hash[SHA256_HASH_SIZE];
    cudaMemcpy(host_block_hash, d_block_hash, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
    host_block_hash[SHA256_HASH_SIZE - 1] = '\0';
    logMessage("Block hash: %s", host_block_hash);

    printResult(host_block_hash, nonce, seconds);

    // Free device memory
    cudaFree(d_block_content);
    cudaFree(d_block_hash);
    cudaFree(d_nonce_result);

    return 0;
}