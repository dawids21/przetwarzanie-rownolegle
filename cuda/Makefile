CUDA_HOME := /usr/local/cuda

NVCC := $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS := -g -G

CUDA_TEST := mandelbrot
CUDA_TEST_SRC := mandelbrot.cu

all: $(CUDA_TEST)

$(CUDA_TEST): $(CUDA_TEST_SRC)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f $(CUDA_TEST)
