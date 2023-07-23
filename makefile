CUDADIR ?= /gpfs/runtime/opt/cuda/11.1.1/cuda
INC=-I${CUDADIR}/include
LIBDIR=-L${CUDADIR}/lib64
LIB=-lcudart

OPENMP = -fopenmp

NVCCFLAGS  = -O3  -Xcompiler "-fPIC -Wall -Wno-unused-function -Wno-strict-aliasing" -std=c++11
NVCCFLAGS += -gencode arch=compute_70,code=sm_70
NVCCFLAGS += -gencode arch=compute_80,code=sm_80

toed: main.o cpu_toed.o gpu_convolve.o gpu_nms.o form_curvelet_main.o
	g++ ${INC} ${OPENMP} -o TOED main.o cpu_toed.o gpu_convolve.o gpu_nms.o form_curvelet_main.o ${LIBDIR} ${LIB}

main.o: main.cpp
	g++ ${OPENMP} -c main.cpp -O3 -o main.o

cpu_toed.o: cpu_toed.cpp
	g++ ${OPENMP} -c cpu_toed.cpp -O3 -o cpu_toed.o

#form_curvelet_main.o: ./curvelet/form_curvelet_main.cpp ./curvelet/curvelet.cpp ./curvelet/curveletmap.cpp ./curvelet/CC_curve_model_3d.cpp ./curvelet/form_curvelet_process.cpp 
#	g++ ${OPENMP} -c ./curvelet/form_curvelet_main.cpp ./curvelet/curvelet.cpp ./curvelet/curveletmap.cpp ./curvelet/CC_curve_model_3d.cpp ./curvelet/form_curvelet_process.cpp -O3 -o form_curvelet_main.o


gpu_convolve.o: gpu_convolve.cu
	nvcc ${INC} ${NVCCFLAGS} -c gpu_convolve.cu -o gpu_convolve.o

gpu_nms.o: gpu_nms.cu
	nvcc ${INC} ${NVCCFLAGS} -c gpu_nms.cu -o gpu_nms.o

form_curvelet_main.o: ./curvelet/form_curvelet_main.cpp
	g++ ${OPENMP} -c ./curvelet/form_curvelet_main.cpp -O3 -o form_curvelet_main.o

clean:
	rm -f toed *.o
