#ifndef CPU_TOED_CPP
#define CPU_TOED_CPP

#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <string.h>
#include <vector>

#include "indices.hpp"
#include <omp.h>

#if OPENCV_SUPPORT
#include <opencv2/opencv.hpp>
#endif

#include "cpu_toed.hpp"

// ==================================== Constructor ===================================
// Define parameters used by functions in the class and allocate 2d arrays dynamically
// ====================================================================================
template<typename T>
ThirdOrderEdgeDetectionCPU<T>::ThirdOrderEdgeDetectionCPU(int H, int W, int sigma, int kernel_size, int cpu_nthreads) {
    img_height = H;
    img_width = W;
    output_dir = "./output_files";

    kernel_sz = kernel_size;
    shifted_kernel_sz = kernel_sz + 2;
    g_sig = sigma;

    // -- interpolated img size --
    interp_img_height = img_height*2;
    interp_img_width  = img_width*2;

    // openmp threads
    omp_threads = cpu_nthreads;
    img            = new T[img_height*img_width];

    // -- interpolated image map --
    Ix             = new T[interp_img_height*interp_img_width];
    Iy             = new T[interp_img_height*interp_img_width];
    I_grad_mag     = new T[interp_img_height*interp_img_width];
    I_orient       = new T[interp_img_height*interp_img_width];  

    // -- subpixel position map --
    subpix_pos_x_map        = new T[interp_img_height*interp_img_width];
    subpix_pos_y_map        = new T[interp_img_height*interp_img_width];
    subpix_grad_mag_map     = new T[interp_img_height*interp_img_width];

    // -- number of data for each edge: subpix x and y, orientation, TO_grad_mag --
    num_of_edge_data = 4;
    subpix_edge_pts_final   = new T[interp_img_height*interp_img_width*num_of_edge_data];
}

// ========================= preprocessing ==========================
// Initialize 2d arrays, with OpenCV supported
// ==================================================================
#if OPENCV_SUPPORT
template<typename T>
void ThirdOrderEdgeDetectionCPU<T>::preprocessing(cv::Mat image) {
    
    // -- input img initialization --
    for (int i = 0; i < img_height; i++) {
        for (int j = 0; j < img_width; j++) {
            img(i, j) = (double)image.at<uchar>(i, j);
        }
    }

    // -- interpolated img initialization --
    for (int i = 0; i < interp_img_height; i++) {
        for (int j = 0; j < interp_img_width; j++) {
            Ix(i, j)         = 0;
            Iy(i, j)         = 0;
            I_grad_mag(i,j)  = 0;
            I_orient(i,j)    = 0;      

            subpix_pos_x_map(i, j)       = 0;
            subpix_pos_y_map(i, j)       = 0;
            subpix_grad_mag_map(i, j)    = 0;
        }
    }

    for (int i = 0; i < interp_img_height*interp_img_width; i++) {
        for (int j = 0; j < num_of_edge_data; j++) {
            subpix_edge_pts_final(i, j)  = 0;
        }
    }
}
#endif

// ========================= preprocessing ==========================
// Initialize 2d arrays, without OpenCV supported
// ==================================================================
template<typename T>
void ThirdOrderEdgeDetectionCPU<T>::preprocessing(std::ifstream& scan_infile) {
    
    // -- input img initialization --
    for (int i = 0; i < img_height; i++) {
        for (int j = 0; j < img_width; j++) {
            img(i, j) = (int)scan_infile.get();
        }
    }

    // -- interpolated img initialization --
    for (int i = 0; i < interp_img_height; i++) {
        for (int j = 0; j < interp_img_width; j++) {
            Ix(i, j)         = 0;
            Iy(i, j)         = 0;
            I_grad_mag(i,j)  = 0;
            I_orient(i,j)    = 0;      

            subpix_pos_x_map(i, j)       = 0;
            subpix_pos_y_map(i, j)       = 0;
            subpix_grad_mag_map(i, j)    = 0;
        }
    }

    for (int i = 0; i < interp_img_height*interp_img_width; i++) {
        for (int j = 0; j < num_of_edge_data; j++) {
            subpix_edge_pts_final(i, j)  = 0;
        }
    }
}

template<typename T>
void ThirdOrderEdgeDetectionCPU<T>::convolve_img()
{
    const int cent = (kernel_sz-1)/2;
    const int cent_interp = cent+1;

    // 1D Gaussian and its derivatives at scale g_sig (toed_cfg::sigma).
    // Unshifted taps are centered at 0; half-pixel taps are shifted by 0.5.
    // Length is shifted_kernel_sz = kernel_size + 2.
    const T sig = static_cast<T>(g_sig);
    const T sig2 = sig * sig;
    const T inv_sqrt_2pi = static_cast<T>(1) / std::sqrt(static_cast<T>(2) * PI);
    const T sig3 = sig2 * sig;
    const T sig5 = sig2 * sig3;
    const T sig7 = sig2 * sig5;

    auto fill_1d = [&](T shift, std::vector<T>& G, std::vector<T>& dG,
                       std::vector<T>& d2G, std::vector<T>& d3G) {
        for (int p = -cent_interp; p <= cent_interp; ++p) {
            const int idx = p + cent_interp;
            const T x = static_cast<T>(p) + shift;
            const T e = std::exp(-(x * x) / (static_cast<T>(2) * sig2)) * inv_sqrt_2pi;
            G[idx]   = e / sig;
            dG[idx]  = (-x) * e / sig3;
            d2G[idx] = (x * x - sig2) * e / sig5;
            d3G[idx] = (x * (static_cast<T>(3) * sig2 - x * x)) * e / sig7;
        }
    };

    std::vector<T> Gx(shifted_kernel_sz), G_of_x(shifted_kernel_sz);
    std::vector<T> Gxx(shifted_kernel_sz), Gxxx(shifted_kernel_sz);
    std::vector<T> Gx_sh(shifted_kernel_sz), G_of_x_sh(shifted_kernel_sz);
    std::vector<T> Gxx_sh(shifted_kernel_sz), Gxxx_sh(shifted_kernel_sz);

    fill_1d(static_cast<T>(0),   G_of_x,    Gx,    Gxx,    Gxxx);
    fill_1d(static_cast<T>(0.5), G_of_x_sh, Gx_sh, Gxx_sh, Gxxx_sh);

	// -- do convolution and compute gradient magnitude --
    omp_set_num_threads(omp_threads);
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        T TO_conv_Ix, TO_conv_Iy;
        T TO_conv_mag;

        T fx;
        T fy;
        T fxx;
        T fyy;
        T fxy;
        T fxxy;
        T fxyy;
        T fxxx;
        T fyyy;

        #pragma omp for schedule(dynamic)
        // -- do convolution --
        for (int i = 0; i < img_height; i++) {
            for (int j = 0; j < img_width; j++) {
                int si = i*2;
                int sj = j*2;

                fx = 0;
                fy = 0;
                fxx = 0;
                fyy = 0;
                fxy = 0;
                fxxy = 0;
                fxyy = 0;
                fxxx = 0;
                fyyy = 0;
                
                // -- 1) loop over the 17x17 filter --
                for (int p = -cent; p <= cent; p++) {
                    for (int q = -cent; q <= cent; q++) {
                        if ((i-p) < 0 || (j-q) < 0 || (i-p) >= img_height || (j-q) >= img_width)
                            continue;

                        fx += img(i-p, j-q) * (Gx[q+cent+1]     * G_of_x[p+cent+1]);      // Gx * G_of_y
                        fy += img(i-p, j-q) * (G_of_x[q+cent+1] *     Gx[p+cent+1]);      // G_of_x * Gy

                        fxx  += img(i-p, j-q) * Gxx[q+cent+1]    * G_of_x[p+cent+1];    // Gxx * G_of_y
                        fxy  += img(i-p, j-q) * Gx[q+cent+1]     * Gx[p+cent+1];        // Gx * Gy
                        fyy  += img(i-p, j-q) * G_of_x[q+cent+1] * Gxx[p+cent+1];       // G_of_x * Gyy
                        fxxy += img(i-p, j-q) * Gxx[q+cent+1]    * Gx[p+cent+1];        // Gxx * Gy
                        fxyy += img(i-p, j-q) * Gx[q+cent+1]     * Gxx[p+cent+1];       // Gx * Gyy
                        fxxx += img(i-p, j-q) * Gxxx[q+cent+1]   * G_of_x[p+cent+1];    // Gxxx * G_of_y
                        fyyy += img(i-p, j-q) * G_of_x[q+cent+1] * Gxxx[p+cent+1];      // G_of_x * Gyyy
                    }
                }              

                Ix(si,sj) = fx;
                Iy(si,sj) = fy;
                I_grad_mag(si, sj) = std::sqrt(fx*fx + fy*fy);

                TO_conv_Ix = fx * (2*fxx*fxx + 2*fxy*fxy) + fy * (2*fxx*fxy + 2*fyy*fxy) + 2*fx*fy*fxxy + fy*fy*fxyy + fx*fx*fxxx;
                TO_conv_Iy = fx * (2*fxx*fxy + 2*fyy*fxy) + fy * (2*fyy*fyy + 2*fxy*fxy) + 2*fx*fy*fxyy + fx*fx*fxxy + fy*fy*fyyy;
                TO_conv_mag = std::sqrt( TO_conv_Ix *TO_conv_Ix + TO_conv_Iy * TO_conv_Iy );
                TO_conv_Ix /= TO_conv_mag;
                TO_conv_Iy /= TO_conv_mag;
                I_orient(si, sj) = std::atan2(TO_conv_Ix, -TO_conv_Iy);
                // ---------------------------------------------------------

                fx = 0;
                fy = 0;
                fxx = 0;
                fyy = 0;
                fxy = 0;
                fxxy = 0;
                fxyy = 0;
                fxxx = 0;
                fyyy = 0;

                // -- 2) loop over the 19x19 filter, right top, shifted in x only --
                for (int p = -cent_interp; p <= cent_interp; p++) {
                    for (int q = -cent_interp; q <= cent_interp; q++) {
                        if ((i-p) < 0 || (j-q) < 0 || (i-p) >= img_height || (j-q) >= img_width)
                            continue;

                        fx += img(i-p, j-q) * Gx_sh[q+cent_interp]     * G_of_x[p+cent_interp];
                        fy += img(i-p, j-q) * G_of_x_sh[q+cent_interp] *     Gx[p+cent_interp];

                        fxx  += img(i-p, j-q) * Gxx_sh[q+cent_interp]    * G_of_x[p+cent_interp];
                        fxy  += img(i-p, j-q) * Gx_sh[q+cent_interp]     * Gx[p+cent_interp];
                        fyy  += img(i-p, j-q) * G_of_x_sh[q+cent_interp] * Gxx[p+cent_interp];
                        fxxy += img(i-p, j-q) * Gxx_sh[q+cent_interp]    * Gx[p+cent_interp];
                        fxyy += img(i-p, j-q) * Gx_sh[q+cent_interp]     * Gxx[p+cent_interp];
                        fxxx += img(i-p, j-q) * Gxxx_sh[q+cent_interp]   * G_of_x[p+cent_interp];
                        fyyy += img(i-p, j-q) * G_of_x_sh[q+cent_interp] * Gxxx[p+cent_interp];
                    }
                }

                Ix(si,sj+1) = fx;
                Iy(si,sj+1) = fy;
                I_grad_mag(si, sj+1) = std::sqrt(fx*fx + fy*fy);

                TO_conv_Ix = fx * (2*fxx*fxx + 2*fxy*fxy) + fy * (2*fxx*fxy + 2*fyy*fxy) + 2*fx*fy*fxxy + fy*fy*fxyy + fx*fx*fxxx;
                TO_conv_Iy = fx * (2*fxx*fxy + 2*fyy*fxy) + fy * (2*fyy*fyy + 2*fxy*fxy) + 2*fx*fy*fxyy + fx*fx*fxxy + fy*fy*fyyy;
                TO_conv_mag = std::sqrt( TO_conv_Ix *TO_conv_Ix + TO_conv_Iy * TO_conv_Iy );
                TO_conv_Ix /= TO_conv_mag;
                TO_conv_Iy /= TO_conv_mag;
                I_orient(si, sj+1) = std::atan2(TO_conv_Ix, -TO_conv_Iy);
                // ----------------------------------------------------------------

                fx = 0;
                fy = 0;
                fxx = 0;
                fyy = 0;
                fxy = 0;
                fxxy = 0;
                fxyy = 0;
                fxxx = 0;
                fyyy = 0;

                // -- 3) loop over the 19x19 filter, left bottom, shifted in y only --
                for (int p = -cent_interp; p <= cent_interp; p++) {
                    for (int q = -cent_interp; q <= cent_interp; q++) {
                        if ((i-p) < 0 || (j-q) < 0 || (i-p) >= img_height || (j-q) >= img_width)
                            continue;

                        fx += img(i-p, j-q) * Gx[q+cent_interp]     * G_of_x_sh[p+cent_interp];      // Gx * G_of_y
                        fy += img(i-p, j-q) * G_of_x[q+cent_interp] *     Gx_sh[p+cent_interp];      // G_of_x * Gy

                        fxx  += img(i-p, j-q) * Gxx[q+cent_interp]    * G_of_x_sh[p+cent_interp];    // Gxx * G_of_y
                        fxy  += img(i-p, j-q) * Gx[q+cent_interp]     * Gx_sh[p+cent_interp];        // Gx * Gy
                        fyy  += img(i-p, j-q) * G_of_x[q+cent_interp] * Gxx_sh[p+cent_interp];       // G_of_x * Gyy
                        fxxy += img(i-p, j-q) * Gxx[q+cent_interp]    * Gx_sh[p+cent_interp];        // Gxx * Gy
                        fxyy += img(i-p, j-q) * Gx[q+cent_interp]     * Gxx_sh[p+cent_interp];       // Gx * Gyy
                        fxxx += img(i-p, j-q) * Gxxx[q+cent_interp]   * G_of_x_sh[p+cent_interp];    // Gxxx * G_of_y
                        fyyy += img(i-p, j-q) * G_of_x[q+cent_interp] * Gxxx_sh[p+cent_interp];      // G_of_x * Gyyy
                    }
                }

                Ix(si+1,sj) = fx;
                Iy(si+1,sj) = fy;
                I_grad_mag(si+1, sj) = std::sqrt(fx*fx + fy*fy);

                TO_conv_Ix = fx * (2*fxx*fxx + 2*fxy*fxy) + fy * (2*fxx*fxy + 2*fyy*fxy) + 2*fx*fy*fxxy + fy*fy*fxyy + fx*fx*fxxx;
                TO_conv_Iy = fx * (2*fxx*fxy + 2*fyy*fxy) + fy * (2*fyy*fyy + 2*fxy*fxy) + 2*fx*fy*fxyy + fx*fx*fxxy + fy*fy*fyyy;
                TO_conv_mag = std::sqrt( TO_conv_Ix *TO_conv_Ix + TO_conv_Iy * TO_conv_Iy );
                TO_conv_Ix /= TO_conv_mag;
                TO_conv_Iy /= TO_conv_mag;
                I_orient(si+1, sj) = std::atan2(TO_conv_Ix, -TO_conv_Iy);
                // ----------------------------------------------------------------

                fx = 0;
                fy = 0;
                fxx = 0;
                fyy = 0;
                fxy = 0;
                fxxy = 0;
                fxyy = 0;
                fxxx = 0;
                fyyy = 0;

                // -- 4) loop over the 19x19 filter, right bottom, shifted in both x and y --
                for (int p = -cent_interp; p <= cent_interp; p++) {
                    for (int q = -cent_interp; q <= cent_interp; q++) {
                        if ((i-p) < 0 || (j-q) < 0 || (i-p) >= img_height || (j-q) >= img_width)
                            continue;

                        fx += img(i-p, j-q) * Gx_sh[q+cent_interp]     * G_of_x_sh[p+cent_interp];      // Gx * G_of_y
                        fy += img(i-p, j-q) * G_of_x_sh[q+cent_interp] *     Gx_sh[p+cent_interp];      // G_of_x * Gy

                        fxx  += img(i-p, j-q) * Gxx_sh[q+cent_interp]    * G_of_x_sh[p+cent_interp];    // Gxx * G_of_y
                        fxy  += img(i-p, j-q) * Gx_sh[q+cent_interp]     * Gx_sh[p+cent_interp];        // Gx * Gy
                        fyy  += img(i-p, j-q) * G_of_x_sh[q+cent_interp] * Gxx_sh[p+cent_interp];       // G_of_x * Gyy
                        fxxy += img(i-p, j-q) * Gxx_sh[q+cent_interp]    * Gx_sh[p+cent_interp];        // Gxx * Gy
                        fxyy += img(i-p, j-q) * Gx_sh[q+cent_interp]     * Gxx_sh[p+cent_interp];       // Gx * Gyy
                        fxxx += img(i-p, j-q) * Gxxx_sh[q+cent_interp]   * G_of_x_sh[p+cent_interp];    // Gxxx * G_of_y
                        fyyy += img(i-p, j-q) * G_of_x_sh[q+cent_interp] * Gxxx_sh[p+cent_interp];      // G_of_x * Gyyy
                    }
                }

                Ix(si+1,sj+1) = fx;
                Iy(si+1,sj+1) = fy;
                I_grad_mag(si+1, sj+1) = std::sqrt(fx*fx + fy*fy);

                TO_conv_Ix = fx * (2*fxx*fxx + 2*fxy*fxy) + fy * (2*fxx*fxy + 2*fyy*fxy) + 2*fx*fy*fxxy + fy*fy*fxyy + fx*fx*fxxx;
                TO_conv_Iy = fx * (2*fxx*fxy + 2*fyy*fxy) + fy * (2*fyy*fyy + 2*fxy*fxy) + 2*fx*fy*fxyy + fx*fx*fxxy + fy*fy*fyyy;
                TO_conv_mag = std::sqrt( TO_conv_Ix *TO_conv_Ix + TO_conv_Iy * TO_conv_Iy );
                TO_conv_Ix /= TO_conv_mag;
                TO_conv_Iy /= TO_conv_mag;
                I_orient(si+1, sj+1) = std::atan2(TO_conv_Ix, -TO_conv_Iy);
            }
        }
    }
    double test_time = omp_get_wtime() - start;
    std::cout<<"- Time of image convolution (OpenMP): "<<test_time*1000<<" (ms)"<<std::endl;
    time_conv = test_time;

    #if WriteDataToFile
    write_array_to_file("Ix_cpu.txt", Ix, interp_img_height, interp_img_width);
    write_array_to_file("Iy_cpu.txt", Iy, interp_img_height, interp_img_width);
    write_array_to_file("I_grad_mag_cpu.txt", I_grad_mag, interp_img_height, interp_img_width);
    write_array_to_file("I_orient_cpu.txt", I_orient, interp_img_height, interp_img_width);
    #endif
}

// ======================================== Non-maximal Suppression (NMS) ============================================
// (1) Decide the quadrant the gradient belongs to by looking at the signs and size of gradients in x and y directions
// (2) Points which magnitude are greater than both it's neighbors in the direction of their gradients (slope) are
//     considered as peaks.
// (3) Find the subpixel of the edge point by fitting a parabola. This comes from:
//     R. B. Fisher and D. K. Naidu, “A comparison of algorithms for subpixel peak detection,” in Image Technology,
//     Advances in Image Processing, Multimedia and Machine Vis., Berlin, Germany:Springer, 1996, pp. 385–404.
// ====================================================================================================================
template<typename T>
int ThirdOrderEdgeDetectionCPU<T>::non_maximum_suppresion(T* TOED_edges)
{
    /*T norm_dir_x, norm_dir_y;
    T slope, fp, fm;
    T coeff_A, coeff_B, coeff_C, s, s_star;
    T max_f, subpix_grad_x, subpix_grad_y;
    T subpix_grad_mag;*/
    const int sn = 1;

    omp_set_num_threads(omp_threads);
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        T norm_dir_x, norm_dir_y;
        T slope, fp, fm;
        T coeff_A, coeff_B, coeff_C, s, s_star;
        T max_f, subpix_grad_x, subpix_grad_y;
        T subpix_grad_mag;

        #pragma omp for schedule(dynamic)
        for (int j = toed_cfg::nms_border; j < interp_img_width - toed_cfg::nms_border; j+=sn) {
            for (int i = toed_cfg::nms_border; i < interp_img_height - toed_cfg::nms_border; i+=sn) {
                // -- ignore neglectable gradient magnitude --
                if (I_grad_mag(i, j) <= toed_cfg::grad_mag_thresh)
                    continue;

                // -- ignore invalid gradient direction --
                if ((std::abs(Ix(i, j)) < 10e-6) && (std::abs(Iy(i, j)) < 10e-6))
                    continue;

                // -- calculate the unit direction --
                norm_dir_x = Ix(i,j) / I_grad_mag(i,j);
                norm_dir_y = Iy(i,j) / I_grad_mag(i,j);

                // -- find corresponding quadrant --
                if ((Ix(i,j) >= 0) && (Iy(i,j) >= 0)) {
                    if (Ix(i,j) >= Iy(i,j)) {         // -- 1st quadrant --
                        slope = norm_dir_y / norm_dir_x;
                        fp = I_grad_mag(i, j+sn) * (1-slope) + I_grad_mag(i+sn, j+sn) * slope;
                        fm = I_grad_mag(i, j-sn) * (1-slope) + I_grad_mag(i-sn, j-sn) * slope;
                    }
                    else {                              // -- 2nd quadrant --
                        slope = norm_dir_x / norm_dir_y;
                        fp = I_grad_mag(i+sn, j) * (1-slope) + I_grad_mag(i+sn, j+sn) * slope;
                        fm = I_grad_mag(i-sn, j) * (1-slope) + I_grad_mag(i-sn, j-sn) * slope;
                    }
                }
                else if ((Ix(i,j) < 0) && (Iy(i,j) >= 0)) {
                    if (abs(Ix(i,j)) < Iy(i,j)) {     // -- 3rd quadrant --
                        slope = -norm_dir_x / norm_dir_y;
                        fp = I_grad_mag(i+sn, j) * (1-slope) + I_grad_mag(i+sn, j-sn) * slope;
                        fm = I_grad_mag(i-sn, j) * (1-slope) + I_grad_mag(i-sn, j+sn)  * slope;
                    }
                    else {                              // -- 4th quadrant --
                        slope = -norm_dir_y / norm_dir_x;
                        fp = I_grad_mag(i, j-sn) * (1-slope) + I_grad_mag(i+sn, j-sn) * slope;
                        fm = I_grad_mag(i, j+sn) * (1-slope) + I_grad_mag(i-sn, j+sn) * slope;
                    }
                }
                else if ((Ix(i,j) < 0) && (Iy(i,j) < 0)) {
                    if(abs(Ix(i,j)) >= abs(Iy(i,j))) {            // -- 5th quadrant --
                        slope = norm_dir_y / norm_dir_x;
                        fp = I_grad_mag(i, j-sn) * (1-slope) + I_grad_mag(i-sn, j-sn) * slope;
                        fm = I_grad_mag(i, j+sn) * (1-slope) + I_grad_mag(i+sn, j+sn) * slope;
                    }
                    else {                              // -- 6th quadrant --
                        slope = norm_dir_x / norm_dir_y;
                        fp = I_grad_mag(i-sn, j) * (1-slope) + I_grad_mag(i-sn, j-sn) * slope;
                        fm = I_grad_mag(i+sn, j) * (1-slope) + I_grad_mag(i+sn, j+sn) * slope;
                    }
                }
                else if ((Ix(i,j) >= 0) && (Iy(i,j) < 0)) {
                    if(Ix(i,j) < abs(Iy(i,j))) {      // -- 7th quadrant --
                        slope = -norm_dir_x / norm_dir_y;
                        fp = I_grad_mag(i-sn, j) * (1-slope) + I_grad_mag(i-sn, j+sn) * slope;
                        fm = I_grad_mag(i+sn, j) * (1-slope) + I_grad_mag(i+sn, j-sn) * slope;
                    }
                    else {                              // -- 8th quadrant --
                        slope = -norm_dir_y / norm_dir_x;
                        fp = I_grad_mag(i, j+sn) * (1-slope) + I_grad_mag(i-sn, j+sn) * slope;
                        fm = I_grad_mag(i, j-sn) * (1-slope) + I_grad_mag(i+sn, j-sn) * slope;
                    }
                }

                // -- fit a parabola to find the edge subpixel location when doing max test --
                s = std::sqrt(1+slope*slope);
                if((I_grad_mag(i, j) >  fm && I_grad_mag(i, j) > fp) ||  // -- abs max --
                   (I_grad_mag(i, j) >  fm && I_grad_mag(i, j) >= fp) || // -- relaxed max --
                (I_grad_mag(i, j) >= fm && I_grad_mag(i, j) >  fp)) {

                    // -- fit a parabola; define coefficients --
                    coeff_A = (fm+fp-2*I_grad_mag(i, j))/(2*s*s);
                    coeff_B = (fp-fm)/(2*s);
                    coeff_C = I_grad_mag(i, j);

                    s_star = -coeff_B/(2*coeff_A); // -- location of max --
                    max_f = coeff_A*s_star*s_star + coeff_B*s_star + coeff_C; // -- value of max --

                    if(abs(s_star) <= std::sqrt(2)) { // -- significant max is within a pixel --

                        // -- subpixel magnitude in x and y --
                        subpix_grad_x = max_f*norm_dir_x;
                        subpix_grad_y = max_f*norm_dir_y;

                        // -- subpixel gradient magnitude --
                        subpix_grad_mag = std::sqrt(subpix_grad_x*subpix_grad_x + subpix_grad_y*subpix_grad_y);

                        // store subpixel positions in coordinates maps
                        subpix_pos_x_map(i, j) = j + s_star * norm_dir_x;
                        subpix_pos_y_map(i, j) = i + s_star * norm_dir_y;

                        // TODO:
                        // -- store gradient magnitude of subpixel edge in the map --
                        subpix_grad_mag_map(i, j) = subpix_grad_mag;
                    }
                }
            }
        }
    }
    double end = omp_get_wtime() - start;
    std::cout<<"- Time of NMS (OpenMP): "<<end*1000<<" (ms)"<<std::endl;
    time_nms = end;

    #if WriteDataToFile
    write_array_to_file("subpix_pos_x_map_cpu.txt", subpix_pos_x_map, interp_img_height, interp_img_width);
    write_array_to_file("subpix_pos_y_map_cpu.txt", subpix_pos_y_map, interp_img_height, interp_img_width);
    #endif

    // construct edge maps 
    // -- loop over the subpix_pos_x_map to push to an output list --
    edge_pt_list_idx = 0;
    for (int i = 0; i < interp_img_height; i++) {
        for (int j = 0; j < interp_img_width; j++) {
            if (subpix_pos_x_map(i, j) != 0) {
                // -- store all necessary information of final edges --
                // -- 1) subpixel location x --
                subpix_edge_pts_final(edge_pt_list_idx, 0) = (subpix_pos_x_map(i, j)-1) / 2;
                TOED_edges(edge_pt_list_idx, 0) = (subpix_pos_x_map(i, j)-1) / 2;

                // -- 2) subpixel location y --
                subpix_edge_pts_final(edge_pt_list_idx, 1) = (subpix_pos_y_map(i, j)-1) / 2;
                TOED_edges(edge_pt_list_idx, 1) = (subpix_pos_y_map(i, j)-1) / 2;

                // -- 3) orientation of subpixel --
                subpix_edge_pts_final(edge_pt_list_idx, 2) = I_orient(i, j);
                TOED_edges(edge_pt_list_idx, 2) = I_orient(i, j);

                // -- 4) subpixel gradient magnitude --
                subpix_edge_pts_final(edge_pt_list_idx, 3) = subpix_grad_mag_map(i, j);
                TOED_edges(edge_pt_list_idx, 3) = subpix_grad_mag_map(i, j);

                // -- 5) add up the edge point list index --
                edge_pt_list_idx++;
            }
            else {
                continue;
            }
        }
    }

    //#if WriteDataToFile
    write_array_to_file("data_final_output_cpu.txt", subpix_edge_pts_final, edge_pt_list_idx, num_of_edge_data);
    //#endif

    return edge_pt_list_idx;
}

template<typename T>
void ThirdOrderEdgeDetectionCPU<T>::set_output_dir(const std::string& dir)
{
    output_dir = dir.empty() ? "./output_files" : dir;
}

// ===================================== Write data to file for debugging =======================================
// Writes a 2d dybamically allocated array to a text file for debugging
// ==============================================================================================================
template<typename T>
void ThirdOrderEdgeDetectionCPU<T>::write_array_to_file(std::string filename, T *wr_data, int first_dim, int second_dim)
{
#define wr_data(i, j) wr_data[(i) * second_dim + (j)]

    std::cout<<"writing data to a file "<<filename<<" ..."<<std::endl;
    std::string out_file_name = output_dir;
    if (!out_file_name.empty() && out_file_name.back() != '/')
        out_file_name.push_back('/');
    out_file_name.append(filename);
	std::ofstream out_file;
    out_file.open(out_file_name);
    if ( !out_file.is_open() )
      std::cout<<"write data file cannot be opened!"<<std::endl;

	for (int i = 0; i < first_dim; i++) {
		for (int j = 0; j < second_dim; j++) {
			out_file << wr_data(i, j) <<"\t";
		}
		out_file << "\n";
	}

    out_file.close();
#undef wr_data
}

// ===================================== Read data from file for debugging ======================================
// Reads data for debugging
// ==============================================================================================================
template<typename T>
void ThirdOrderEdgeDetectionCPU<T>::read_array_from_file(std::string filename, T *rd_data, int first_dim, int second_dim)
{
#define rd_data(i, j) rd_data[(i) * second_dim + (j)]
    std::cout<<"reading data from a file "<<filename<<std::endl;
    std::string in_file_name = "./test_files/";
    in_file_name.append(filename);
    std::fstream in_file;
    T data;
    int j = 0, i = 0;

    in_file.open(in_file_name, std::ios_base::in);
    if (!in_file) {
        std::cerr << "input read file not existed!\n";
    }
    else {
        while (in_file >> data) {
            rd_data(i, j) = data;
            j++;
            if (j == second_dim) {
                j = 0;
                i++;
            }
        }
    }
#undef rd_data
}

// ===================================== Destructor =======================================
// Free all the 2d dynamic arrays allocated in the constructor
// ========================================================================================
template<typename T>
ThirdOrderEdgeDetectionCPU<T>::~ThirdOrderEdgeDetectionCPU () {
    // free memory
    delete[] img;
    delete[] Ix;
    delete[] Iy;
    delete[] I_grad_mag;
    delete[] I_orient;

    delete[] subpix_pos_x_map;
    delete[] subpix_pos_y_map;
    delete[] subpix_grad_mag_map;

    delete[] subpix_edge_pts_final;
}

// Explicit template instantiations
template class ThirdOrderEdgeDetectionCPU<double>;
template class ThirdOrderEdgeDetectionCPU<float>;

#endif    // TODE_CPP