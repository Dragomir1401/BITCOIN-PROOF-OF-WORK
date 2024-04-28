#ifndef UTILS_H
#define UTILS_H

#include "sha256.h"

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

extern BYTE expected_hashed_tx1[];
extern BYTE expected_hashed_tx2[];
extern BYTE expected_hashed_tx3[];
extern BYTE expected_hashed_tx4[];
extern BYTE expected_tx2_hashed_100_times[];
extern BYTE expected_tx2_hashed_50_times[];
extern BYTE expected_hashed_tx12[];
extern BYTE expected_hashed_tx34[];
extern BYTE expected_top_hash[];
extern BYTE expected_block_hash[];

#define DIFFICULTY difficulty_5_zeros
#define MAX_NONCE UINT32_MAX

void apply_sha256(const BYTE *tx, size_t tx_length, BYTE *hashed_tx, int n);
int compare_hashes(BYTE* hash1, BYTE* hash2);

#endif // UTILS_H
