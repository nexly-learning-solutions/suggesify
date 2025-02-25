CPP ?= /lib/cpp
CC = mpiCC
NVCC = nvcc
LOAD = mpiCC

BUILD_DIR ?= $(shell pwd)/build
PREFIX ?= $(shell pwd)/recs
CUDA_INCLUDE_DIR ?= /usr/local/cuda/include
MPI_INCLUDE_DIR ?= /usr/lib/openmpi/include
JSONCPP_INCLUDE_DIR ?= /usr/include/jsoncpp
INCLUDE_DIRS = \
    -I$(CUDA_INCLUDE_DIR) \
    -I$(MPI_INCLUDE_DIR) \
    -I$(JSONCPP_INCLUDE_DIR) \
    -I$(BUILD_DIR)/include

LIBRARY_PATHS = \
    -L/usr/lib/atlas-base \
    -L/usr/local/cuda/lib64 \
    -L/usr/local/lib/
LIBRARIES = \
    -lcudnn \
    -lcurand \
    -lcublas \
    -lcudart \
    -ljsoncpp \
    -lnetcdf_c++4 \
    -lnetcdf \
    -lblas \
    -ldl \
    -lstdc++

ifdef DEBUG
  CFLAGS = \
      -g \
      -O0 \
      -Wall \
      -std=c++20 \
      -fPIC \
      -DOMPI_SKIP_MPICXX \
      -MMD \
      -MP

  NVCC_FLAGS = \
      -g \
      -O0 \
      --device-debug \
      --generate-line-info \
      -std=c++20 \
      --compiler-options '-fPIC' \
      --compiler-options '-Wall' \
      -use_fast_math \
      --ptxas-options '-v' \
      -gencode arch=compute_80,code=sm_80 \
      -gencode arch=compute_75,code=sm_75 \
      -gencode arch=compute_70,code=sm_70 \
      -DOMPI_SKIP_MPICXX

  $(info ************ DEBUG mode ************)

else
  CFLAGS = \
      -O3 \
      -std=c++20 \
      -fPIC \
      -DOMPI_SKIP_MPICXX \
      -MMD \
      -MP

  NVCC_FLAGS = \
      -O3 \
      -std=c++20 \
      --compiler-options '-fPIC' \
      -use_fast_math \
      --ptxas-options '-v' \
      -gencode arch=compute_80,code=sm_80 \
      -gencode arch=compute_75,code=sm_75 \
      -gencode arch=compute_70,code=sm_70 \
      -DOMPI_SKIP_MPICXX

  $(info ************ RELEASE mode ************)
endif

CPPFLAGS = $(CFLAGS)
NVCCFLAGS = $(NVCC_FLAGS)
INCLUDE_FLAGS = $(INCLUDE_DIRS)
LIBRARY_FLAGS = $(LIBRARY_PATHS) $(LIBRARIES)

check-tools:
	@{ command -v $(NVCC) >/dev/null 2>&1 || { echo "Error: $(NVCC) is not installed. Please install the CUDA Toolkit."; exit 1; } }
	@{ command -v $(CC) >/dev/null 2>&1 || { echo "Error: $(CC) is not installed. Please install the MPI library."; exit 1; } }

%.d: %.cpp
	@$(CPP) -MM $(CPPFLAGS) $(INCLUDE_FLAGS) $< > $@

all: check-tools
	$(MAKE) -C src
	$(MAKE) -C tests

install: all
	mkdir -p $(PREFIX)
	@for dir in lib bin include; do \
		cp -rfp $(BUILD_DIR)/$$dir $(PREFIX); \
	done

run-tests: check-tools
	$(MAKE) -C tests run-tests

clean:
	$(MAKE) -C src clean
	$(MAKE) -C tests clean

.PHONY: all install run-tests clean check-tools