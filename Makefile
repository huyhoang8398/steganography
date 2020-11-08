NVCC = nvcc
CUDAFLAGS = --machine 64
SRC_FILES = main.cu stena_gpu.cu auto_tester.cu ppm_lib.cpp stena_cpu.cpp 

all:
	$(NVCC) $(SRC_FILES) $(CUDAFLAGS) -o stena

clean: 
	rm -rf build

run:
	./stena
