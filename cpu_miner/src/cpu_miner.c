#include <stdio.h>
#include <stdint.h>
#include "../include/utils.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>

// #define DEBUG

// TEST 1. SHA256 basic test: sha256(tx1) and sha256(tx2).
int test1() {
    BYTE hashed_tx[SHA256_HASH_SIZE];

    apply_sha256(tx1, strlen((const char*)tx1), hashed_tx, 1);
    printf("Computed hash of tx1: %s\n", hashed_tx);	
    printf("Expected hash of tx1: %s\n", expected_hashed_tx1);
    printf("Test 1 - sha256(tx1): %s\n", !strcmp((const char*)expected_hashed_tx1, (const char*)hashed_tx) ? "PASS" : "FAILED");

    apply_sha256(tx2, strlen((const char*)tx2), hashed_tx, 1);
    printf("Computed hash of tx2: %s\n", hashed_tx);	
    printf("Expected hash of tx2: %s\n", expected_hashed_tx2);
    printf("Test 1 - sha256(tx2): %s\n", !strcmp((const char*)expected_hashed_tx2, (const char*)hashed_tx) ? "PASS" : "FAILED");

    return 0;
}

// TEST 2. Construct a basic Merkle Tree: sha256(sha256(sha256(tx1)+sha256(tx2))+sha256(sha256(tx3)+sha256(tx4)))
int test2() {
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

    apply_sha256(tx1, strlen((const char*)tx1), hashed_tx1, 1);
    apply_sha256(tx2, strlen((const char*)tx2), hashed_tx2, 1);
    apply_sha256(tx3, strlen((const char*)tx3), hashed_tx3, 1);
    apply_sha256(tx4, strlen((const char*)tx4), hashed_tx4, 1);

    // sha256(tx1)+sha256(tx2)
    strcpy((char *)tx12, (const char *)hashed_tx1);
    strcat((char *)tx12, (const char *)hashed_tx2);

    // sha256(sha256(tx1)+sha256(tx2))
    apply_sha256(tx12, strlen((const char*)tx12), hashed_tx12, 1);
    printf("Test 2 - sha256(sha256(tx1)+sha256(tx2)): %s\n", !strcmp((const char*)expected_hashed_tx12, (const char*)hashed_tx12) ? "PASS" : "FAILED");

    // sha256(tx3)+sha256(tx4)
    strcpy((char *)tx34, (const char *)hashed_tx3);
    strcat((char *)tx34, (const char *)hashed_tx4);

    // sha256(sha256(tx3)+sha256(tx4))
    apply_sha256(tx34, strlen((const char*)tx34), hashed_tx34, 1);
    printf("Test 2 - sha256(sha256(tx3)+sha256(tx4)): %s\n", !strcmp((const char*)expected_hashed_tx34, (const char*)hashed_tx34) ? "PASS" : "FAILED");

    // sha256(sha256(tx1)+sha256(tx2))+sha256(sha256(tx3)+sha256(tx4))
    strcpy((char *)tx1234, (const char *)hashed_tx12);
    strcat((char *)tx1234, (const char *)hashed_tx34);

    // sha256(sha256(sha256(tx1)+sha256(tx2))+sha256(sha256(tx3)+sha256(tx4)))
    apply_sha256(tx1234, strlen((const char*)tx34), top_hash, 1);
    printf("Test 2 - Top Hash: %s\n", !strcmp((const char*)expected_top_hash, (const char*)top_hash) ? "PASS" : "FAILED");

    return 0;
}

// TEST 3: Apply sha256 n times on tx2: sha256 o sha256 o ... o sha256(tx2)
int test3() {
    BYTE tx2_hashed_100_times[SHA256_HASH_SIZE];
    apply_sha256(tx2, strlen((const char*)tx2), tx2_hashed_100_times, 100);
    printf("Test 3 - sha256 of tx2 100 times: %s\n", !strcmp((const char*)expected_tx2_hashed_100_times, (const char*)tx2_hashed_100_times) ? "PASS" : "FAILED");

    BYTE tx2_hashed_50_times[SHA256_HASH_SIZE];
    apply_sha256(tx2, strlen((const char*)tx2), tx2_hashed_50_times, 50);
    printf("Test 3 - sha256 of tx2 50 times: %s\n", !strcmp((const char*)expected_tx2_hashed_50_times, (const char*)tx2_hashed_50_times) ? "PASS" : "FAILED");

    return 0;
}

// TEST 4: Compute block hash of tx1,tx2,tx3,tx4, with a given nonce: sha256(prev_block_hash + top_hash + nonce)
int test4() {
    BYTE hashed_tx1[SHA256_HASH_SIZE], hashed_tx2[SHA256_HASH_SIZE], hashed_tx3[SHA256_HASH_SIZE], hashed_tx4[SHA256_HASH_SIZE],
         tx12[SHA256_HASH_SIZE * 2], tx34[SHA256_HASH_SIZE * 2], hashed_tx12[SHA256_HASH_SIZE], hashed_tx34[SHA256_HASH_SIZE],
         tx1234[SHA256_HASH_SIZE * 2], top_hash[SHA256_HASH_SIZE], block_content[BLOCK_SIZE], block_hash[SHA256_HASH_SIZE];
    char nonce_string[NONCE_SIZE];
    uint32_t nonce = 3246176006;

    // top_hash
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

    // prev_block_hash + top_hash + nonce
    strcpy((char*)block_content, (const char*)prev_block_hash);
    strcat((char*)block_content, (const char*)top_hash);
    sprintf(nonce_string, "%u", nonce);
    strcat((char*)block_content, nonce_string);

    // sha256(prev_block_hash + top_hash + nonce)
    apply_sha256(block_content, strlen((const char*)block_content), block_hash, 1);

    printf("Test 4 - block_hash: %s\n", !strcmp((const char*)expected_block_hash, (const char*)block_hash) ? "PASS" : "FAILED");
    return 0;
}

// TEST 5: Compute block hash of tx1,tx2,tx3,tx4, but find nonce so that the block hash is below a given difficulty
int test5() {
    BYTE hashed_tx1[SHA256_HASH_SIZE], hashed_tx2[SHA256_HASH_SIZE], hashed_tx3[SHA256_HASH_SIZE], hashed_tx4[SHA256_HASH_SIZE],
         tx12[SHA256_HASH_SIZE * 2], tx34[SHA256_HASH_SIZE * 2], hashed_tx12[SHA256_HASH_SIZE], hashed_tx34[SHA256_HASH_SIZE],
         tx1234[SHA256_HASH_SIZE * 2], top_hash[SHA256_HASH_SIZE], block_content[BLOCK_SIZE], block_hash[SHA256_HASH_SIZE];
    char nonce_string[NONCE_SIZE];
    uint32_t nonce;
    size_t current_length;
    time_t startTime, endTime;

    // top_hash
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

    // prev_block_hash + top_hash
    strcpy((char*)block_content, (const char*)prev_block_hash);
    strcat((char*)block_content, (const char*)top_hash);

    current_length = strlen((char*) block_content);
    printf("Block content without nonce: %s\n", block_content);

    printf("Started searching nonces from 1 to %lu\n", (long unsigned) MAX_NONCE);
    startTime = time(NULL);
    for (nonce = 0; nonce <= MAX_NONCE; nonce++) {
        sprintf(nonce_string, "%u", nonce);
        strcpy((char*) block_content + current_length, nonce_string); // Overwrite previous nonce
        apply_sha256(block_content, strlen((const char*)block_content), block_hash, 1);

        #ifdef DEBUG
        printf("Computed hash for nonce %s: %s\n", nonce_string, block_hash);
        #endif

        if (compare_hashes(block_hash, DIFFICULTY) <= 0) {
            break;
        }
    }
    endTime = time(NULL);
    double duration = difftime(endTime, startTime);

    if (compare_hashes(block_hash, DIFFICULTY) <= 0) {
        printf("Hash found: %s, lower than difficulty %s, for nonce %u\n", block_hash, DIFFICULTY, nonce);
    } else {
        printf("Hash not found :(\n");
    }

    printf("Event started at: %s", ctime(&startTime));
    printf("Event ended at: %s", ctime(&endTime));
    printf("Execution time: %.2f seconds\n", duration);

    return 0;
}

int main(int argc, char **argv) {
    test1();
    printf("\n");

    test2();
    printf("\n");

    test3();
    printf("\n");

    test4();
    printf("\n");

    test5();
    printf("\n");

    return 0;
}
