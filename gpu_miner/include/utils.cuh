/**********************************************************************

!!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!

*********************************************************************/
#ifndef UTILS_H
#define UTILS_H

#include "sha256.cuh"
#include <stdint.h>

/********************** PRINT RESULT **************************/

void printResult(BYTE *block_hash,  uint64_t nonce, float seconds);

/*************************** TIME *****************************/

void startTiming(cudaEvent_t *start, cudaEvent_t *stop);
float stopTiming(cudaEvent_t *start, cudaEvent_t *stop);

/*************************** SHA256 ***************************/
#define SHA256_HASH_SIZE 65 // 32 byte array is 64 characters long + 1 for null terminator
#define NONCE_SIZE 11 // UINT32_MAX is 10 chars long + 1 for null terminator
#define BLOCK_SIZE (2 * SHA256_HASH_SIZE + NONCE_SIZE + 1) // prev_hash + txs_hash + nonce + 1 for null terminator

extern BYTE tx1[];
extern BYTE tx2[];
extern BYTE tx3[];
extern BYTE tx4[];
extern BYTE prev_block_hash[];
extern BYTE difficulty_5_zeros[];

#define DIFFICULTY difficulty_5_zeros
#define MAX_NONCE 1e8

__host__ __device__ size_t d_strlen(const char *str);
__host__ __device__ void d_strcpy(char *dest, const char *src);
__host__ __device__ void apply_sha256(const BYTE *tx, size_t tx_length, BYTE *hashed_tx, int n);
__device__ int intToString(uint64_t num, char* out);
__device__ int compare_hashes(BYTE* hash1, BYTE* hash2);

#endif // UTILS_H
