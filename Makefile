# Simple alternative build path (no CMake). Works on Linux/WSL directly;
# on native Windows, run this from a "x64 Native Tools" prompt so nvcc can
# find cl.exe, or just use CMakeLists.txt instead.

NVCC      ?= nvcc
ARCHES    ?= 60 70 75 80 86 89 90 120
GENCODE   := $(foreach a,$(ARCHES),-gencode arch=compute_$(a),code=sm_$(a))

INCLUDES  := -Iinclude -Isrc -Isrc/cuda
NVCCFLAGS := -O3 -std=c++17 $(GENCODE) $(INCLUDES)
LIBS      := -lcufft

SRCS_CU  := src/main.cu src/cuda/chirp_generation.cu src/cuda/matched_filter.cu \
            src/cuda/range_doppler.cu src/cuda/coherent_integration.cu src/cuda/peak_detection.cu
SRCS_CPP := src/scenario.cpp src/cpu_reference/cpu_pipeline.cpp

TARGET := radar_cuda

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRCS_CU) $(SRCS_CPP)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIBS)

clean:
	rm -f $(TARGET) $(TARGET).exe *.o
