MODULE_ROOT = $(shell pwd)

BUILD_DIR ?= $(shell pwd)/../build
BIN_DIR = $(BUILD_DIR)/tests/bin
TEST_SOURCES_DIR = bitfusion
OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/tests

SOURCES=$(shell find '$(TEST_SOURCES_DIR)' -type f -name '*.cpp')
OBJECTS := $(SOURCES:$(TEST_SOURCES_DIR)/%.cpp=$(OBJS_BUILD_DIR)/%.o)

DEP := $(OBJECTS:.o=.d)

BITFUSION_SRC_DIR = ../src
LIB_BITFUSION = $(BUILD_DIR)/lib/libbitfusion.a
LIB_BITFUSION_UTILS = $(BUILD_DIR)/lib/libbitfusion_utils.so
 
INCLUDES = \
	-isystem /usr/local/cuda/include \
	-isystem /usr/include/jsoncpp \
	-isystem /usr/lib/openmpi/include \
	-isystem /usr/lib/openmpi/include/openmpi \
	-I../src

LIB = \
	$(BUILD_DIR)/lib \
	/usr/local/cuda/lib64 \
	/usr/lib/openmpi/lib

LIBS ?= $(LIB:%=-L%)

LLIB = \
	cudnn \
	curand \
	cublas \
	cudart \
	jsoncpp \
	netcdf \
	netcdf_c++4 \
	blas \
	dl \
	stdc++ \
	mpi_cxx \
	mpi \
	cppunit \
	bitfusion_utils 

LOAD_LIBS = $(LLIB:%=-l%)

CC = g++
CFLAGS ?=-Wall -std=c++20 -pthread 
LDFLAGS ?= -Wl,-rpath=/usr/local/cuda/lib64

all: $(BIN_DIR)/unittests

$(LIB_BITFUSION):
	cd $(BITFUSION_SRC_DIR)/engine && make

$(LIB_BITFUSION_UTILS):
	cd $(BITFUSION_SRC_DIR)/utils && make

$(BIN_DIR)/unittests: $(LIB_BUTFUSION) $(LIB_BITFUSION_UTILS) $(OBJECTS)
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $(LIBS) unittests.cpp $^ $(LIB_BITFUSION) -o $@ $(LOAD_LIBS)

run-tests: $(BIN_DIR)/unittests 
	LD_LIBRARY_PATH=$(BUILD_DIR)/lib $(BIN_DIR)/unittests

clean:
	rm -rf $(BIN_DIR)	

$(OBJS_BUILD_DIR)/%.o: $(TEST_SOURCES_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP $(INCLUDES) -c $< -o $@

