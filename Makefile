# =============================================================================
# Makefile - Enterprise Multi-Stream Spectral Signal Pipeline (cuFFT) - Windows
# =============================================================================

# Use the environment variable CUDA_PATH if set, otherwise default to common Windows install path
CUDA_PATH   ?= C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6
NVCC        := "$(CUDA_PATH)/bin/nvcc.exe"
CXX_FLAGS   := -std=c++17 -O3
NVCC_FLAGS  := $(CXX_FLAGS) -Xcompiler -Wall
ARCH        := -gencode arch=compute_89,code=sm_89
CUDA_LIBS   := -lcufft -lcudart

SRC_DIR     := src
BIN_DIR     := bin
TARGET      := $(BIN_DIR)/spectral_pipeline.exe

SRCS        := src/spectral_pipeline.cu

.PHONY: all clean dirs

all: dirs $(TARGET)

dirs:
	@if not exist bin mkdir bin
	@if not exist data mkdir data
	@if not exist data\input mkdir data\input
	@if not exist data\output mkdir data\output
	@if not exist docs mkdir docs

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $^ -o $@ $(CUDA_LIBS)
	@echo Build successful: $@

clean:
	@if exist $(BIN_DIR) rd /s /q $(BIN_DIR)
	@if exist data\output rd /s /q data\output
