#!/bin/bash

# Local directory containing the source files
LOCAL_DIR="/home/student/Documents/ASC/Teme/asc-public/assignments/2-cuda_proof_of_work/gpu_miner/src"

# Remote SSH username and host
REMOTE_USER="andrei.dragomir1401"
REMOTE_HOST="fep.grid.pub.ro"

# Remote directory where the files will be copied
REMOTE_DIR="/export/home/acs/stud/a/andrei.dragomir1401/asc-public/assignments/2-cuda_proof_of_work/gpu_miner"

# Copy files from the local directory to the remote directory
scp -r "${LOCAL_DIR}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

echo "Files have been transferred successfully."
