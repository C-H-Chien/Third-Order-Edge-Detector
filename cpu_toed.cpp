#ifndef CPU_TOED_CPP
#define CPU_TOED_CPP

#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>
#include <opencv2/opencv.hpp>

#include "indices.hpp"
#include <omp.h>

#include "cpu_toed.hpp"

// ==================================== Constructor ===================================
// Define parameters used by functions in the class and allocate 2d arrays dynamically
// ====================================================================================
template<typename T>
ThirdOrderEdgeDetectionCPU<T>::ThirdOrderEdgeDetectionCPU(int H, int W, int sigma, int kernel_size, int cpu_nthreads) {
    img_height = H;
    img_width = W;

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
// Initialize 2d arrays
// ==================================================================
template<typename T>
void ThirdOrderEdgeDetectionCPU<T>::preprocessing(cv::Mat image) {
    
    // -- input img initialization --
    for (int i = 0; i < img_height; i++) {
        for (int j = 0; j < img_width; j++) {
            // img(i, j) = (int)scan_infile.get();
            img(i, j) = (double)image.at<uchar>(i, j);
        }
    }

    std::cout << img(0,0) << "\t" << img(0,1) << "\t" << img(0,2) << std::endl;

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

    // -- 2 types of filters
    //T Gxxx[shifted_kernel_sz],    Gxx[shifted_kernel_sz],    Gx[shifted_kernel_sz],    G_of_x[shifted_kernel_sz];
    //T Gxxx_sh[shifted_kernel_sz], Gxx_sh[shifted_kernel_sz], Gx_sh[shifted_kernel_sz], G_of_x_sh[shifted_kernel_sz];
    //T Gxxx[shifted_kernel_sz],    Gxx[shifted_kernel_sz];
    //T Gxxx_sh[shifted_kernel_sz], Gxx_sh[shifted_kernel_sz];

    // -- 1D convolution filter, part I --
    /*T dx = 0;
    T dy = 0;
    for (int p = -cent_interp; p <= cent_interp; p++) {
        Gxxx[p+cent_interp]   = (((p+dy)*(3*g_sig*g_sig-(p+dy)*(p+dy)))*std::exp(-(p+dy)*(p+dy)/(2*g_sig*g_sig)))/(std::sqrt(2*PI)*g_sig*g_sig*g_sig*g_sig*g_sig*g_sig*g_sig);
        Gxx[p+cent_interp]    = (((p+dy)*(p+dy)-g_sig*g_sig)*std::exp(-(p+dy)*(p+dy)/(2*g_sig*g_sig)))/(std::sqrt(2*PI)*g_sig*g_sig*g_sig*g_sig*g_sig);
        //Gx[p+cent_interp]     = (-(p+dy)*std::exp(-(p+dy)*(p+dy)/(2*g_sig*g_sig)))/(std::sqrt(2*PI)*g_sig*g_sig*g_sig);
        //G_of_x[p+cent_interp] = std::exp(-(p+dx)*(p+dx)/(2*g_sig*g_sig))/(std::sqrt(2*PI)*g_sig);        
    }*/
    
    T Gx[] = {1.79817087452687e-05,	0.000133830225764885,	0.000763597358165040,	0.00332388630895351,	0.0109551878084803,	0.0269954832565940,	0.0485690983747094,	0.0604926811297858,	0.0440081658455374,	0,	-0.0440081658455374,	-0.0604926811297858,	-0.0485690983747094,	-0.0269954832565940,	-0.0109551878084803,	-0.00332388630895351,	-0.000763597358165040,	-0.000133830225764885,	-1.79817087452687e-05};
    T G_of_x[] = {7.99187055345274e-06,   6.69151128824427e-05,   0.000436341347522880,   0.00221592420596900,    0.00876415024678427,    0.0269954832565940, 0.0647587978329459, 0.120985362259572,  0.176032663382150,  0.199471140200716,  0.176032663382150,  0.120985362259572,  0.0647587978329459, 0.0269954832565940, 0.00876415024678427,    0.00221592420596900,    0.000436341347522880,   6.69151128824427e-05,   7.99187055345274e-06};
    T Gxx[] = {3.84608770384913e-05,	0.000250931673309160,	0.00122721003990810,	0.00443184841193801,	0.0115029471989044,	0.0202466124424455,	0.0202371243227956,	0,	-0.0330061243841531,	-0.0498677850501791,	-0.0330061243841531,	0,	0.0202371243227956,	0.0202466124424455,	0.0115029471989044,	0.00443184841193801,	0.00122721003990810,	0.000250931673309160,	3.84608770384913e-05};
    T Gxxx[] = {7.75461189639711e-05,	0.000434948233735878,	0.00176581889075666,	0.00498582946343026,	0.00890109009439027,	0.00674887081414851,	-0.00910670594525801,	-0.0302463405648929,	-0.0302556140188070,	0,	0.0302556140188070,	0.0302463405648929,	0.00910670594525801,	-0.00674887081414851,	-0.00890109009439027,	-0.00498582946343026,	-0.00176581889075666,	-0.000434948233735878,	-7.75461189639711e-05};
    
    /*dx = 0.5;
    dy = 0.5;
    for (int p = -cent_interp; p <= cent_interp; p++) {
        Gxxx_sh[p+cent_interp]   = (((p+dy)*(3*g_sig*g_sig-(p+dy)*(p+dy)))*std::exp(-(p+dy)*(p+dy)/(2*g_sig*g_sig)))/(std::sqrt(2*PI)*g_sig*g_sig*g_sig*g_sig*g_sig*g_sig*g_sig);
        Gxx_sh[p+cent_interp]    = (((p+dy)*(p+dy)-g_sig*g_sig)*std::exp(-(p+dy)*(p+dy)/(2*g_sig*g_sig)))/(std::sqrt(2*PI)*g_sig*g_sig*g_sig*g_sig*g_sig);
        //Gx_sh[p+cent_interp]     = (-(p+dy)*std::exp(-(p+dy)*(p+dy)/(2*g_sig*g_sig)))/(std::sqrt(2*PI)*g_sig*g_sig*g_sig);
        //G_of_x_sh[p+cent_interp] = std::exp(-(p+dx)*(p+dx)/(2*g_sig*g_sig))/(std::sqrt(2*PI)*g_sig);
    }*/

    T G_of_x_sh[] = {2.38593182706025e-05,	0.000176297841183723,	0.00101452402864988,	0.00454678125079553,	0.0158698259178337,	0.0431386594132558,	0.0913245426945110,	0.150568716077402,	0.193334058401425,	0.193334058401425,	0.150568716077402,	0.0913245426945110,	0.0431386594132558,	0.0158698259178337,	0.00454678125079553,	0.00101452402864988,	0.000176297841183723,	2.38593182706025e-05,	2.51475364429622e-06};
    T Gx_sh[] = {5.07010513250303e-05,	0.000330558452219480,	0.00164860154655606,	0.00625182421984385,	0.0178535541575629,	0.0377463269865988,	0.0570778391840694,	0.0564632685290258,	0.0241667573001781,	-0.0241667573001781,	-0.0564632685290258,	-0.0570778391840694,	-0.0377463269865988,	-0.0178535541575629,	-0.00625182421984385,	-0.00164860154655606,	-0.000330558452219480,	-5.07010513250303e-05,	-5.97253990520353e-06};
    T Gxx_sh[] = {0.000101774904498039,	0.000575722637615595,	0.00242534650599113,	0.00745956298958641,	0.0161177919477999,	0.0222433712599600,	0.0128425138164156,	-0.0164684533209659,	-0.0453126699378339,	-0.0453126699378339,	-0.0164684533209659,	0.0128425138164156,	0.0222433712599600,	0.0161177919477999,	0.00745956298958641,	0.00242534650599113,	0.000575722637615595,	0.000101774904498039,	1.35560938637843e-05};
    T Gxxx_sh[] = {0.000190921146395817,	0.000914200719419500,	0.00311688729895755,	0.00713098700075939,	0.00920573886249338,	0.000589786359165606,	-0.0205123484567749,	-0.0344073042598751,	-0.0177474623923183,	0.0177474623923183,	0.0344073042598751,	0.0205123484567749,	-0.000589786359165606,	-0.00920573886249338,	-0.00713098700075939,	-0.00311688729895755,	-0.000914200719419500,	-0.000190921146395817,	-2.92094529738860e-05};

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
        for (int j = 10; j < interp_img_width - 10; j+=sn) {
            for (int i = 10; i < interp_img_height - 10; i+=sn) {
                // -- ignore neglectable gradient magnitude --
                if (I_grad_mag(i, j) <= 2)
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

// ===================================== Write data to file for debugging =======================================
// Writes a 2d dybamically allocated array to a text file for debugging
// ==============================================================================================================
template<typename T>
void ThirdOrderEdgeDetectionCPU<T>::write_array_to_file(std::string filename, T *wr_data, int first_dim, int second_dim)
{
#define wr_data(i, j) wr_data[(i) * second_dim + (j)]

    std::cout<<"writing data to a file "<<filename<<" ..."<<std::endl;
    std::string out_file_name = "./output_files/";
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

#endif    // TODE_CPP