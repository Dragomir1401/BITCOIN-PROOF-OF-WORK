# !!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!  DO NOT MODIFY THIS FILE !!!!!!!!!!!!!!!!!!!!!!

COMPILER = nvcc
ARCH =
CFLAGS = -Iinclude -dc $(ARCH) -Xcompiler -Wall
SRC_DIR = src
OBJS = $(SRC_DIR)/gpu_miner.o $(SRC_DIR)/sha256.o $(SRC_DIR)/utils.o
EXEC = gpu_miner
LIBS = -lm

ifdef LOCAL
build: $(EXEC)

$(EXEC): $(OBJS)
	$(COMPILER) $(ARCH) $(OBJS) -o $(EXEC) $(LIBS)

$(SRC_DIR)/gpu_miner.o: $(SRC_DIR)/gpu_miner.cu $(SRC_DIR)/utils.o
	$(COMPILER) $(CFLAGS) -c $< -o $@

$(SRC_DIR)/sha256.o: $(SRC_DIR)/sha256.cu
	$(COMPILER) $(CFLAGS) -c $< -o $@

$(SRC_DIR)/utils.o: $(SRC_DIR)/utils.cu $(SRC_DIR)/sha256.o
	$(COMPILER) $(CFLAGS) -c $< -o $@

run: $(EXEC)
	./$(EXEC)

.PHONY: all build run clean
endif

include ../Makefile
