OPENCVDIR ?= /gpfs/data/bkimia/cchien3/opencv_install_CH_test
INC=-I${OPENCVDIR}/include/opencv4
LIBDIR=-L${OPENCVDIR}/lib64
LIB=-lopencv_core -lopencv_imgcodecs -lopencv_highgui
OPENMP = -fopenmp

toed: main_cpu_dp.o cpu_toed.o 
	g++ ${INC} ${OPENMP} -o TOED main_cpu_dp.o cpu_toed.o ${LIBDIR} ${LIB}

main_cpu_dp.o: main_cpu_DP.cpp
	g++ ${INC} ${OPENMP} -c main_cpu_DP.cpp -O3 -o main_cpu_dp.o

cpu_toed.o: cpu_toed.cpp
	g++ ${INC} ${OPENMP} -c cpu_toed.cpp -O3 -o cpu_toed.o

clean:
	rm -f toed *.o
