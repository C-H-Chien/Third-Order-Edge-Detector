# Standalone CPU/OpenMP build for subpixel third-order correction.
# Does not require main_cpu_DP, main_gpu_cpu, or the TOED detector.
#
#   cd subpix_TO_correction
#   make
#   ./TO_correct <edgemap> [orientation] [thresh] [sigma] [nthreads] [output_dir]

OPENCVDIR ?= /oscar/rt/sw/external/opencv/4.6.0
INC=-I${OPENCVDIR}/include/opencv4 -I. -I..
LIBDIR=-L${OPENCVDIR}/lib64
LIB=-lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui
OPENMP = -fopenmp
CXX ?= g++
CXXFLAGS = -O3 ${OPENMP} ${INC}

.PHONY: all clean

all: TO_correct

TO_correct: main_subpix_to_correction.o subpix_to_correction.o
	${CXX} ${OPENMP} -o TO_correct main_subpix_to_correction.o subpix_to_correction.o ${LIBDIR} ${LIB}

main_subpix_to_correction.o: main_subpix_to_correction.cpp subpix_to_correction.hpp
	${CXX} ${CXXFLAGS} -c main_subpix_to_correction.cpp -o main_subpix_to_correction.o

subpix_to_correction.o: subpix_to_correction.cpp subpix_to_correction.hpp
	${CXX} ${CXXFLAGS} -c subpix_to_correction.cpp -o subpix_to_correction.o

clean:
	rm -f TO_correct *.o
