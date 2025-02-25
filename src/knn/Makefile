# Shell to use
SHELL := /bin/sh

# Include additional makefile if it exists
-include ../Makefile.inc

# Compiler settings
CC := g++
NVCC := nvcc
CFLAGS := -Wall -Werror
CUDA_FLAGS := -std=c++20 -arch=sm_80 -Xcompiler -Wall,-Werror,-fPIC,-fopenmp

# Directories
BUILD_DIR := build
SOURCES_DIR := .
HEADERS_DIR := $(SOURCES_DIR)
HEADERS_BUILD_DIR := $(BUILD_DIR)/include/knn
OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/knn/cpp
CU_OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/knn/cuda
LIB_BUILD_DIR := $(BUILD_DIR)/lib

# File sources
HEADERS := $(wildcard $(HEADERS_DIR)/*.h $(HEADERS_DIR)/*.hpp)
CPP_SOURCES := $(wildcard $(SOURCES_DIR)/*.c $(SOURCES_DIR)/*.cc $(SOURCES_DIR)/*.cpp)
CU_SOURCES := $(wildcard $(SOURCES_DIR)/*.cu)
OBJECTS := $(CPP_SOURCES:$(SOURCES_DIR)/%.cpp=$(OBJS_BUILD_DIR)/%.o)
CU_OBJECTS := $(CU_SOURCES:$(SOURCES_DIR)/%.cu=$(CU_OBJS_BUILD_DIR)/%.o)

# Dependency files
DEP := $(OBJECTS:.o=.d)
CU_DEP := $(CU_OBJECTS:.o=.d)

# CUDA includes and libraries
CUDA_INCLUDES := -I../utils
LDFLAGS := -L$(LIB_BUILD_DIR)
CUDA_LIBS := -lcudart -lcublas -lcudnn -lcurand -lcusparse -lcufft
CUDA_LOADLIBS := -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs

# Default target
all: $(LIB_BUILD_DIR)/libbitfusion_knn.so

# Build shared library
$(LIB_BUILD_DIR)/libbitfusion_knn.so: $(OBJECTS) $(CU_OBJECTS) | directories
	@echo "Linking shared library $@..."
	$(CC) -shared $(LDFLAGS) $(CUDA_LIBS) $^ -o $@ $(CUDA_LOADLIBS)
	@mkdir -p $(HEADERS_BUILD_DIR)
	cp $(HEADERS) $(HEADERS_BUILD_DIR)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(OBJS_BUILD_DIR)/*.o $(OBJS_BUILD_DIR)/*.d
	rm -f $(CU_OBJS_BUILD_DIR)/*.o $(CU_OBJS_BUILD_DIR)/*.d
	rm -rf $(HEADERS_BUILD_DIR) $(LIB_BUILD_DIR)

# Full clean
distclean: clean
	@echo "Performing full clean..."
	rm -rf $(BUILD_DIR)

# Compile C++ sources into object files
$(OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cpp
	@mkdir -p $(@D)
	@echo "Compiling C++ source $<..."
	$(CC) $(CFLAGS) $(CUDA_INCLUDES) -MMD -MP -c $< -o $@

# Compile CUDA sources into object files
$(CU_OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cu
	@mkdir -p $(@D)
	@echo "Compiling CUDA source $<..."
	$(NVCC) $(CUDA_FLAGS) --keep-dir $(CU_OBJS_BUILD_DIR) $(CUDA_INCLUDES) -MMD -MP -c $< -o $@

# Generate dependency files for CUDA sources
$(CU_OBJS_BUILD_DIR)/%.d: $(SOURCES_DIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) -E -std=c++20 -Xcompiler "-isystem $(CUDA_SYSTEM_INCLUDE_DIR) -MP,-MM" $(CUDA_INCLUDES) $< -o $@

# Include dependency files
-include $(DEP)
-include $(CU_DEP)

# Ensure necessary directories exist
directories:
	@mkdir -p $(OBJS_BUILD_DIR) $(CU_OBJS_BUILD_DIR) $(LIB_BUILD_DIR) $(HEADERS_BUILD_DIR)