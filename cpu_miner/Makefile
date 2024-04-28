COMPILER = gcc
CFLAGS = -Iinclude -Wall
SRC_DIR = src
OBJS = $(SRC_DIR)/cpu_miner.o $(SRC_DIR)/sha256.o $(SRC_DIR)/utils.o
EXEC = cpu_miner
LIBS = -lm

ifdef LOCAL
build: $(EXEC)

$(EXEC): $(OBJS)
	$(COMPILER) $(OBJS) -o $(EXEC) $(LIBS)

$(SRC_DIR)/cpu_miner.o: $(SRC_DIR)/cpu_miner.c $(SRC_DIR)/utils.o
	$(COMPILER) $(CFLAGS) -c $< -o $@

$(SRC_DIR)/sha256.o: $(SRC_DIR)/sha256.c
	$(COMPILER) $(CFLAGS) -c $< -o $@

$(SRC_DIR)/utils.o: $(SRC_DIR)/utils.c $(SRC_DIR)/sha256.o
	$(COMPILER) $(CFLAGS) -c $< -o $@

run: $(EXEC)
	./$(EXEC)

.PHONY: all build run clean
endif

include ../Makefile
