SHELL=/bin/sh
VPATH=

include ../Makefile.inc

SOURCES_DIR := .

HEADERS := $(wildcard *.h *.hpp)
HEADERS_BUILD_DIR := $(BUILD_DIR)/include

OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/engine/cpp
CU_OBJS_BUILD_DIR := $(BUILD_DIR)/tmp/engine/cuda
$(shell mkdir -p $(OBJS_BUILD_DIR))
$(shell mkdir -p $(CU_OBJS_BUILD_DIR))

SOURCES=$(wildcard *.c *.cc *.cpp)
CU_SOURCES=$(wildcard *.cu)

OBJECTS := $(SOURCES:%.cpp=$(OBJS_BUILD_DIR)/%.o)
CU_OBJECTS := $(CU_SOURCES:%.cu=$(CU_OBJS_BUILD_DIR)/%.o)

DEP := $(OBJECTS:.o=.d)
CU_DEP := $(CU_OBJECTS:.o=.d)

CU_INCLUDES += -I../utils

LIB_BUILD_DIR := $(BUILD_DIR)/lib

all: $(LIB_BUILD_DIR)/libbitfusion.a $(LIB_BUILD_DIR)/libbitfusion.so

$(LIB_BUILD_DIR)/libbitfusion.a: $(OBJECTS) $(CU_OBJECTS)
	mkdir -p $(LIB_BUILD_DIR)
	ar -cvr $@ $^
	mkdir -p $(HEADERS_BUILD_DIR)
	cp $(HEADERS) $(HEADERS_BUILD_DIR)

$(LIB_BUILD_DIR)/libbitfusion.so: $(OBJECTS) $(CU_OBJECTS)
	mkdir -p $(BUILD_DIR)/lib
	$(CC) -shared $(LDFLAGS) $(CU_LIBS) $(OBJECTS) $(CU_OBJECTS) -o $@ $(CU_LOADLIBS)

clean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc*
	rm -rf $(OBJS_BUILD_DIR) $(CU_OBJS_BUILD_DIR) $(HEADERS_BUILD_DIR) $(LIB_BUILD_DIR)/libbitfusion.a

distclean:
	rm -f *cudafe* *.fatbin.* *.fatbin *.ii *.cubin *cu.cpp *.ptx *.cpp?.* *.hash *.o *.d work.pc*
	rm -rf $(OBJS_BUILD_DIR) $(CU_OBJS_BUILD_DIR)


$(OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cpp
	$(CC) $(CFLAGS) $(CU_INCLUDES) -c $< -o $@

$(CU_OBJS_BUILD_DIR)/%.o: $(SOURCES_DIR)/%.cu GpuTypes.h  $(CU_OBJS_BUILD_DIR)/%.d
	$(NVCC) $(CU_FLAGS) --keep-dir $(CU_OBJS_BUILD_DIR) $(CU_INCLUDES) $($*.cu.CU_FLAGS) -c $< -o $@

$(CU_OBJS_BUILD_DIR)/%.d: $(SOURCES_DIR)/%.cu
	$(NVCC) -E -std=c++20 -Xcompiler "-isystem $(CUDA_SYSTEM_INCLUDE_DIR) -MP,-MM" $(CU_INCLUDES)  $< -o $@

-include $(DEP)
-include $(CU_DEP)
