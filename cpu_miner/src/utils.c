#include <stdio.h>
#include "../include/utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/*************************** SHA256 ***********************************/

// ============================== DATA =================================
BYTE tx1[] = "FROM_Alice__TO_Bob__5_BTC";
BYTE tx2[] = "FROM_Charlie__TO_David__9_BTC";
BYTE tx3[] = "FROM_Erin__TO_Frank__1_BTC";
BYTE tx4[] = "FROM_Alice__TO_Frank__3_BTC";
BYTE prev_block_hash[SHA256_HASH_SIZE] = "000000000000000000034158a91c1876f5fc2add1e69641e908956ac9de45b93";

// ============================== RESULTS ==============================
BYTE expected_hashed_tx1[SHA256_HASH_SIZE] = "50293d1d5ce8eae30847cf45cc700a876a5a96b257d1f95cbd5437f7acdefd7c";
BYTE expected_hashed_tx2[SHA256_HASH_SIZE] = "5f7992732d6058789ecac6dca9494a35addd8e0e6985c099fb9160c5c437314f";
BYTE expected_hashed_tx3[SHA256_HASH_SIZE] = "a1bfa4bdfb2ca2347d059508862aa640b5fe0bf721190c49b5195a0306cee59f";
BYTE expected_hashed_tx4[SHA256_HASH_SIZE] = "4ebb57d8bd051491cded098b9e4f17d5c682dec952caf4159540a3f2fcc6d712";
BYTE expected_tx2_hashed_100_times[SHA256_HASH_SIZE] = "a1a52c43c75fca7cc3b831d186182ec96bdd44a004a6887f024e5b0725f3fbcf";
BYTE expected_tx2_hashed_50_times[SHA256_HASH_SIZE] = "a66ea99fc0d59a6328bbc61c72f6488dbce50f574a549f43053e8eeb126d99d3";
BYTE expected_hashed_tx12[SHA256_HASH_SIZE] = "2d49e888cb41425bdf918e74bf65747da8d486a1e6adce55c6c59ca43220f902";
BYTE expected_hashed_tx34[SHA256_HASH_SIZE] = "c42626191545b52838c3c490af46e90d22c16849673a7eac523241dc29c597c2";
BYTE expected_top_hash[SHA256_HASH_SIZE] = "a0c60e1ff17667ec6256f2ba779519b7cbcade0f06215fe9717f24ea7e77e6d9";
BYTE expected_block_hash[SHA256_HASH_SIZE] = "6e7d18275f97c08576b1957410256dfdb1802caafd51e42168794f8fd9242f28";

// Difficulty based on the number of leading 0s.
BYTE difficulty_5_zeros[SHA256_HASH_SIZE] = "0000099999999999999999999999999999999999999999999999999999999999";

// Apply sha256 n times
void apply_sha256(const BYTE *tx, size_t tx_length, BYTE *hashed_tx, int n) {
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];
    const char hex_chars[] = "0123456789abcdef";

    sha256_init(&ctx);
    sha256_update(&ctx, tx, tx_length);
    sha256_final(&ctx, buf);

    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
        hashed_tx[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];  // Extract the high nibble
        hashed_tx[i * 2 + 1] = hex_chars[buf[i] & 0x0F];         // Extract the low nibble
    }
    hashed_tx[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate the string

    if (n >= 2) {
        for (int i = 0; i < n-1; i ++) {
            sha256_init(&ctx);
            sha256_update(&ctx, hashed_tx, SHA256_BLOCK_SIZE * 2);
            sha256_final(&ctx, buf);

            for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
                hashed_tx[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];  // Extract the high nibble
                hashed_tx[i * 2 + 1] = hex_chars[buf[i] & 0x0F];         // Extract the low nibble
            }
            hashed_tx[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate the string
        }
    }
}

// Function to compare two hashes
int compare_hashes(BYTE* hash1, BYTE* hash2) {
    for (int i = 0; i < SHA256_HASH_SIZE; i++) {
        if (hash1[i] < hash2[i]) {
            return -1; // hash1 is lower
        } else if (hash1[i] > hash2[i]) {
            return 1; // hash2 is lower
        }
    }
    return 0; // hashes are equal
}
