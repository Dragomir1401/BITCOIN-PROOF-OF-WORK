Nume: Dragomir Andrei
Grupă: 332CA

# Proof of Work Bitcoding implementation in CUDA

------------------------------------ Structure------------------------------------------------------------

    The project is structured in the following way:

    - gpu_miner
        - include
            -sha256.cuh (header file for the sha256 implementation)
            -utils.cuh (header file for the utils)
        - src
            -sha256.cu (source file for the sha256 implementation)
            -utils.cu (source file for the utils)
            -gpu_miner.cu (source file for the main implementation)


------------------------------------ Implementation-------------------------------------------------------

    The implementation follows the model of a cpu iterative implementation of the proof of work algorithm.
The main difference is that the implementation is done in parallel on the GPU. The main idea is to generate
a nonce capable to generate a hash that has a certain number of leading zeros. The implementation is done
in the following way:

    - The main function hashes texts
    - Then it allocates memory on the GPU for the block hash and the block content
    - It copies the block content to the GPU
    - It Calculates the block content based on the previous hash and the top hash
    - Then it creates the grid and block dimensions
    - It starts the timer and calls the kernel function
    - After the kernel function finishes, it stops the timer and copies the result back to the CPU
    - The final step is to print the result and free the memory

    The kernel function uses an atomic counter to check if another thread has already found the solution.
    The implementation is done in the following way:

    - It first check if another thread has already found the solution using the counter
    - Then it calculates the thread index
    - It converts the thread index to a string as nonce
    - It creates a local block content where it copies the block content to not influence other threads
    - It creates a local block hash where it will store the hash of the block content
    - It checks again if another thread has already found the solution because there is no need to calculate
        the sha256 hash if we already have a solution
    - It calculates the sha256 hash of the block content
    - It checks if the solution was already found to not redo the hash comparison
    - It checks if the hash has the required number of leading zeros
    - If it has, it increments the atomic counter and blocks other threads from
        entering this part of the code to ensure the block hash and nonce result are set by the same thread
    - If the nonce result was not set yet as an extra precaution, it sets the block hash and the nonce result

    The block size is set as 256 threads and the grid size is calculated as follows:

    dim3 gridSize((MAX_NONCE + blockSize.x - 1) / blockSize.x)

    The reason for this is to ensure that the grid size is big enough to cover all the possible nonces based on
the block size. The block size is set as 256 threads because it is a good number for the GPU to work efficiently.

    
-------------------------------------Comments-------------------------------------------------------------

    The implementation is done in a way that is very similar to the CPU implementation. The main difference
is that the memory is allocated on the GPU and the kernel function is called in parallel. The hard part
was to understand how to allocate memory on the GPU and how to correctly apply the hashing function.


-------------------------------------Testing--------------------------------------------------------------

    To test the implementation you can use local CUDA or the cluster using the makefile.


-------------------------------------Results--------------------------------------------------------------

    The hash that is first found is: 00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee
    The nonce that was found is: 515800
    The time it took to find the solution is: under 0.1 seconds

    The results has 5 leading zeros as intended. The hash of the content was tested using an online tool
and it was correct. The time it took to find the solution was very fast, under 0.1 seconds using the cluster.
The time results indicate a good choosing of the block size and grid size for the current problem, as each
thread only searches for one nonce and computes the sha256 hash of the block content only once. The results
are consistent with the expected results and the time is consistently under 0.1 seconds.

-------------------------------------References-----------------------------------------------------------

https://gitlab.cs.pub.ro/asc/asc-public/-/blob/master/assignments/README.example.md
https://gitlab.cs.pub.ro/asc/asc-public/-/tree/master/assignments
https://ocw.cs.pub.ro/courses/asc/teme/tema2

## Git

https://github.com/Dragomir1401/BITCOIN-PROOF-OF-WORK

-------------------------------------Feedback-------------------------------------------------------------

    The project was very interesting and I learned a something about how the proof of work algorithm works.
The assignment itself was quite simple and did not test many of the concepts from the laboratory. However,
it was a good practice for basic CUDA programming. I would have liked to see more complex assignments that
test more of the concepts learned in the laboratory.

    The algorithm itself sound very complex, but the solution is not that complex. I think it is suitable if
the target is to let more students solve it. Was pretty hard to test on the cluster, since a scp script was
needed to copy the files to the cluster, but that is just my problem because I cant run locally.

----------------------------------------------------------------------------------------------------------