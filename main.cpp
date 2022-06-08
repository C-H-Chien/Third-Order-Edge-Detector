// =============================================================================
// (c) LEMS, Brown University
// Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// June 2022 
// =============================================================================
#include <cmath>
#include <fstream>
#include <iterator>
#include <iostream>
#include <string.h>
#include <vector>
#include <stdint.h>

// cpu
#include "cpu_toed.hpp"
#include "cpu_toed.cpp"

// gpu
#include "gpu_toed.hpp"

//------------------------------------------------------------------------------
// TODO: add for CPU-GPU error checking code

//------------------------------------------------------------------------------
int main(int argc, char **argv)
{
	// -- Exit if the input image file doesn't open --
	std::string filename(argv[1]);
	std::ifstream infile(filename, std::ios::binary);
	if (!infile.is_open())
	{
		std::cout << "File " << filename << " not found in directory." << std::endl;
		return 0;
	}

	char type[10];
	int height, width, intensity;
	// -- Storing header information and copying into the new ouput images --
	infile >> type >> width >> height >> intensity;

	// read number of threads if passed through command line
	int nthreads = 1;
	if(argc > 2) {
	    nthreads = atoi( argv[2] );
	}

    int gpu_id = 0;
	if(argc > 3) {
	    gpu_id = atoi( argv[3] );
	}

	cudacheck( cudaSetDevice(gpu_id) );

	// -- define parameters --
	// -- (This could be changed to argv input arguments but now let's make it fixed)
	int kernel_size = 17;
	int sigma = 2;

    // ==================================== CANNY EDGE DETECTOR STARTS HERE ===============================================
    #if ThirdOrderEdgeDetector
    printf("############################################\n");
	printf("##         Double Precision Test          ##\n");
	printf("############################################\n");
    printf("\n ==> CPU Test (OpenMP %d threads)  \n", nthreads);
    printf("============================================\n");
    
    ThirdOrderEdgeDetectionCPU<double> toedCPU_fp64(height, width, sigma, kernel_size, nthreads);
    toedCPU_fp64.preprocessing(infile);
    toedCPU_fp64.convolve_img();
    toedCPU_fp64.non_maximum_suppresion();
    
    // go back to the beginning of the file
    infile.clear();
    infile.seekg(0);
    infile >> type >> width >> height >> intensity;

	printf("\n ==> GPU Test \n");
	printf("=============================================\n");
	ThirdOrderEdgeDetectionGPU<double> toedGPU_fp64(gpu_id, height, width, kernel_size, sigma);  // -- class constructor --
	toedGPU_fp64.preprocessing(infile);            // -- preprocessing: array initialization --
	toedGPU_fp64.convolve_img();           // -- convolve image with Gaussian derivative filter --
    toedGPU_fp64.non_maximum_suppresion();

    // go back to the beginning of the file
    infile.clear();
    infile.seekg(0);
    infile >> type >> width >> height >> intensity;

    printf("############################################\n");
	printf("##         Single Precision Test          ##\n");
	printf("############################################\n");
    printf("\n ==> CPU Test (OpenMP %d threads)  \n", nthreads);
    printf("=============================================\n");
    
    ThirdOrderEdgeDetectionCPU<float> toedCPU_fp32(height, width, sigma, kernel_size, nthreads);
    toedCPU_fp32.preprocessing(infile);
    toedCPU_fp32.convolve_img();
    toedCPU_fp32.non_maximum_suppresion();

    // go back to the beginning of the file
    infile.clear();
    infile.seekg(0);
    infile >> type >> width >> height >> intensity;

	printf("\n ==> GPU Test \n");
	printf("=============================================\n");
	ThirdOrderEdgeDetectionGPU<float> toedGPU_fp32(gpu_id, height, width, kernel_size, sigma);  // -- class constructor --
	toedGPU_fp32.preprocessing(infile);            // -- preprocessing: array initialization --
	toedGPU_fp32.convolve_img();           // -- convolve image with Gaussian derivative filter --
    toedGPU_fp32.non_maximum_suppresion();
    #endif

    return 0;
}
