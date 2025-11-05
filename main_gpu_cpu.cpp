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

#include "indices.hpp"

#if OPENCV_SUPPORT
#include <opencv2/opencv.hpp>
#endif

// cpu
#include "cpu_toed.hpp"

// gpu
#include "gpu_toed.hpp"

// curvelet
#include "./curvelet/form_curvelet_main.hpp"

//------------------------------------------------------------------------------
// TODO: add CPU-GPU error checking code

template<typename T>
void initialize_TOED_edges( T* &TOED_edges, int height, int width ) 
{
    TOED_edges = new T[(2*height)*(2*width)*4];
    // initialization
    for (int i = 0; i < (2*height)*(2*width); i++) {
        for (int j = 0; j < 4; j++) {
            TOED_edges(i, j)  = 0;
        }
    }
}

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

    int height, width;
#if OPENCV_SUPPORT
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if ( img.empty() ) {
        std::cerr << "Error: Failed to load image " << filename << std::endl;
        return 0;
    }
    height = img.rows;
    width  = img.cols;
#else
    char type[10];
	int intensity;
	// -- Storing header information and copying into the new ouput images --
	infile >> type >> width >> height >> intensity;
#endif

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

	// -- define parameters (This could be changed to argv input arguments but now let's make it fixed)
	int kernel_size = 17;
	int sigma = 2;

    // ==================================== THIRD-ORDER EDGE DETECTOR STARTS HERE ===============================================
	int edge_num;
    if ( !Use_Double_Precision && !Use_Single_Precision ) { 
        std::cout << "You must choose either single or double precision in indices.hpp file!" << std::endl; 
        exit(1);
    }

    // MARK: Double Precision
    if (Use_Double_Precision) {
        double *TOED_edges;
        initialize_TOED_edges<double>( TOED_edges, height, width );

        printf("############################################\n");
        printf("##         Double Precision Test          ##\n");
        printf("############################################\n");
        printf("\n ==> CPU Test (OpenMP %d threads)  \n", nthreads);
        printf("============================================\n");
        
        ThirdOrderEdgeDetectionCPU<double> toedCPU_fp64(height, width, sigma, kernel_size, nthreads);
#if OPENCV_SUPPORT
        toedCPU_fp64.preprocessing(img);
#else
        toedCPU_fp64.preprocessing(infile);
#endif
        toedCPU_fp64.convolve_img();
        edge_num = toedCPU_fp64.non_maximum_suppresion(TOED_edges);

        // go back to the beginning of the file
#if OPENCV_SUPPORT
#else
        infile.clear();
        infile.seekg(0);
        infile >> type >> width >> height >> intensity;
#endif

//         printf("\n ==> GPU Test \n");
//         printf("=============================================\n");
//         ThirdOrderEdgeDetectionGPU<double> toedGPU_fp64(gpu_id, height, width, kernel_size, sigma);  // -- class constructor --
// #if OPENCV_SUPPORT
//         toedGPU_fp64.preprocessing(img);
// #else
//         toedGPU_fp64.preprocessing(infile);
// #endif
//         toedGPU_fp64.convolve_img();           // -- convolve image with Gaussian derivative filter --
//         toedGPU_fp64.non_maximum_suppresion();

        //> Double precision allows curve formation
#if CurvelFormation

        if (Use_Double_Precision) {
            // -- settings --
            double nrad = 3.5;
            double gap = 1.5;
            double dx = 0.4;
            double dt = 15;
            double token_len = 1;
            double max_k = 0.3;
            unsigned cvlet_style = 3;
            unsigned max_size_to_group = 7;
            //> when output_type is 0, output the curvelet map
            //  when output_type is 1, output the curve fragment graph
            //  when output_type is 2, output the poly arc map
            unsigned output_type = 0;

            //> convert degree to radian
            dt = (dt / 180) * M_PI;

            arrayi chain;
            arrayd info;

            curvelet_formation( chain, info, height, width, TOED_edges, edge_num, 4, 
                                nrad, gap, dx, dt, token_len, max_k, 
                                cvlet_style, max_size_to_group, output_type);

            std::cout << "chain width and height: " << chain.w() << ", " << chain.h() << std::endl;
            std::cout << "info width and height: " << info.w() << ", " << info.h() << std::endl;
        }
#endif

        delete[] TOED_edges;
    }
    
    // MARK: Single Precision
    if (Use_Single_Precision) {
        float *TOED_edges;
        initialize_TOED_edges<float>( TOED_edges, height, width );

#if OPENCV_SUPPORT
#else
        // go back to the beginning of the file
        infile.clear();
        infile.seekg(0);
        infile >> type >> width >> height >> intensity;
#endif

        printf("############################################\n");
        printf("##         Single Precision Test          ##\n");
        printf("############################################\n");
        printf("\n ==> CPU Test (OpenMP %d threads)  \n", nthreads);
        printf("=============================================\n");
        
        ThirdOrderEdgeDetectionCPU<float> toedCPU_fp32(height, width, sigma, kernel_size, nthreads);
#if OPENCV_SUPPORT
        toedCPU_fp32.preprocessing(img);
#else
        toedCPU_fp32.preprocessing(infile);
#endif
        toedCPU_fp32.convolve_img();
        edge_num = toedCPU_fp32.non_maximum_suppresion(TOED_edges);

#if OPENCV_SUPPORT
#else
        // go back to the beginning of the file
        infile.clear();
        infile.seekg(0);
        infile >> type >> width >> height >> intensity;
#endif

        printf("\n ==> GPU Test \n");
        printf("=============================================\n");
        ThirdOrderEdgeDetectionGPU<float> toedGPU_fp32(gpu_id, height, width, kernel_size, sigma);  // -- class constructor --
#if OPENCV_SUPPORT
        toedGPU_fp32.preprocessing(img);            // -- preprocessing: array initialization --
#else
        toedGPU_fp32.preprocessing(infile);            // -- preprocessing: array initialization --
#endif
        toedGPU_fp32.convolve_img();           // -- convolve image with Gaussian derivative filter --
        toedGPU_fp32.non_maximum_suppresion();

        delete[] TOED_edges;
    }


    return 0;
}
