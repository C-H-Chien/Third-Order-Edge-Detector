CUDADIR ?= /gpfs/runtime/opt/cuda/11.1.1/cuda
INC=-I${CUDADIR}/include
LIBDIR=-L${CUDADIR}/lib64
LIB=-lcudart

OPENMP = -fopenmp

NVCCFLAGS  = -O3  -Xcompiler "-fPIC -Wall -Wno-unused-function -Wno-strict-aliasing" -std=c++11
NVCCFLAGS += -gencode arch=compute_70,code=sm_70
NVCCFLAGS += -gencode arch=compute_80,code=sm_80

toed: main.o cpu_toed.o gpu_convolve.o gpu_nms.o
	g++ ${INC} ${OPENMP} -o TOED main.o cpu_toed.o gpu_convolve.o gpu_nms.o ${LIBDIR} ${LIB}

main.o: main.cpp
	g++ ${OPENMP} -c main.cpp -O3 -o main.o

cpu_toed.o: cpu_toed.cpp
	g++ ${OPENMP} -c cpu_toed.cpp -O3 -o cpu_toed.o

gpu_convolve.o: gpu_convolve.cu
	nvcc ${INC} ${NVCCFLAGS} -c gpu_convolve.cu -o gpu_convolve.o

gpu_nms.o: gpu_nms.cu
	nvcc ${INC} ${NVCCFLAGS} -c gpu_nms.cu -o gpu_nms.o

clean:
	rm -f toed *.o
