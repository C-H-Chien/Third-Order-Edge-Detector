#ifndef CPU_TOED_HPP
#define CPU_TOED_HPP

#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>

#include "indices.hpp"
#include <omp.h>


template<typename T>
class ThirdOrderEdgeDetectionCPU {
    int img_height;
    int img_width;
    int interp_img_height;
    int interp_img_width;
    int kernel_sz;
    int shifted_kernel_sz;
    int g_sig;
    int interp_n;
    const T PI = 3.14159265358979323846;

    T *img;
	  T *Ix, *Iy;
    T *I_grad_mag;
    T *I_orient;

    T *subpix_pos_x_map;         // -- store x of subpixel location --
    T *subpix_pos_y_map;         // -- store y of subpixel location --
    T *subpix_grad_mag_map;      // -- store subpixel gradient magnitude --

  public:

    T *subpix_edge_pts_final;    // -- a list of final edge points with all information --
    int edge_pt_list_idx;
    int num_of_edge_data;
    int omp_threads;

    // timing
    double time_conv, time_nms;

    ThirdOrderEdgeDetectionCPU(int, int, int, int, int);
    ~ThirdOrderEdgeDetectionCPU();

    // -- member functions --
    void preprocessing(std::ifstream& scan_infile);
    void preprocessing(cv::Mat image);
    void convolve_img();
    int non_maximum_suppresion(T* TOED_edges);

    void read_array_from_file(std::string filename, T *rd_data, int first_dim, int second_dim);
    void write_array_to_file(std::string filename, T *wr_data, int first_dim, int second_dim);
};



#endif    // TOED_HPP