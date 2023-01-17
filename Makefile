# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda
# CC compiler options:
CC=g++
CC_FLAGS=-Iinclude -MMD -MP -DCPU -Wall
CC_LIBS=-lm -g 
# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-Iinclude
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

## Project file structure ##
# Source file directory:
SRC_DIR = src
# Object file directory:
OBJ_DIR = obj
# Include header file diretory:
INC_DIR = include

## Make variables ##

# Target executable name:
BIN_DIR = bin
BIN = $(BIN_DIR)/nn

# Object files:
SRC_C := $(wildcard $(SRC_DIR)/*.c)
SRC_CUDA := $(wildcard $(SRC_DIR)/*.cu)
OBJS_C := $(SRC_C:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
OBJS_CUDA := $(SRC_CUDA:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

.PHONY: all clean

all: $(BIN)

## Compile ##
# Link c and CUDA compiled object files to target executable:
$(BIN) : $(OBJS_C) $(OBJS_CUDA) | $(BIN_DIR)
	$(CC) $(CC_FLAGS) $(OBJS_C) $(OBJS_CUDA) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main.c file to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CC_FLAGS) -c $< -o $@ $(CC_LIBS)

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

# Clean objects in object directory.
clean:
	$(RM) -rv bin/nn $(OBJ_DIR)

-include $(OBJ:.o=.d)
