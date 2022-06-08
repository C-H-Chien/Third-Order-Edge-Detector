#ifndef GPU_TOED_HPP
#define GPU_TOED_HPP

#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "gpu_kernels.hpp"
#include "indices.hpp"

template<typename T>
class ThirdOrderEdgeDetectionGPU {
    int device_id;
    int img_height;
    int img_width;
    int interp_img_height;
    int interp_img_width;
    int kernel_sz;
    int shifted_kernel_sz;
    int g_sig;
    const T PI = 3.14159265358979323846;

    T *img, *dev_img;
    T *Ix, *dev_Ix; 
    T *Iy, *dev_Iy;
    T *I_grad_mag, *dev_I_grad_mag;
    T *I_orient, *dev_I_orient;
    T *dev_Gx,    *dev_Gxx,    *dev_Gxxx,    *dev_G_of_x;
    T *dev_Gx_sh, *dev_Gxx_sh, *dev_Gxxx_sh, *dev_G_of_x_sh;

    // NMS
    T *subpix_pos_x_map, *dev_subpix_pos_x_map;         // -- store x of subpixel location --
    T *subpix_pos_y_map, *dev_subpix_pos_y_map;         // -- store y of subpixel location --

  public:

    int edge_pt_list_idx;
    int num_of_edge_data;

    // timing
    float time_conv, time_nms;
    cudaEvent_t start, stop;

    ThirdOrderEdgeDetectionGPU(int, int, int, int, int);
    ~ThirdOrderEdgeDetectionGPU();

    // -- member functions --
    void preprocessing(std::ifstream& scan_infile);
    void convolve_img();
    void non_maximum_suppresion();

    void read_array_from_file(std::string filename, T *rd_data, int first_dim, int second_dim);
    void write_array_to_file(std::string filename, T *wr_data, int first_dim, int second_dim);
};

// ==================================== Constructor ===================================
// Define parameters used by functions in the class and allocate 2d arrays dynamically
// ====================================================================================
template<typename T>
ThirdOrderEdgeDetectionGPU<T>::ThirdOrderEdgeDetectionGPU (int device, int H, int W, int kernel_size, int sigma) {
    device_id = device;
    img_height = H;
    img_width = W;

    kernel_sz = kernel_size;
    shifted_kernel_sz = kernel_sz + 2;
    g_sig = sigma;

    // -- interpolated img size --
    interp_img_height = img_height*2;
    interp_img_width  = img_width*2;

    // cpu
    img            = new T[img_height*img_width];
    Ix             = new T[interp_img_height*interp_img_width];
    Iy             = new T[interp_img_height*interp_img_width];
    I_grad_mag     = new T[interp_img_height*interp_img_width];
    I_orient       = new T[interp_img_height*interp_img_width]; 

    subpix_pos_x_map   = new T[interp_img_height*interp_img_width];
    subpix_pos_y_map   = new T[interp_img_height*interp_img_width]; 

    // --------------------------------------------------------------------------------
    // gpu
    cudacheck( cudaMalloc((void**)&dev_img,         img_height*img_width*sizeof(T)) );
    cudacheck( cudaMalloc((void**)&dev_Ix,          interp_img_height*interp_img_width*sizeof(T)) );
    cudacheck( cudaMalloc((void**)&dev_Iy,          interp_img_height*interp_img_width*sizeof(T)) );
    cudacheck( cudaMalloc((void**)&dev_I_grad_mag,  interp_img_height*interp_img_width*sizeof(T)) );
    cudacheck( cudaMalloc((void**)&dev_I_orient,    interp_img_height*interp_img_width*sizeof(T)) );

    cudacheck( cudaMalloc((void**)&dev_Gx,          shifted_kernel_sz *sizeof(T)) );
    cudacheck( cudaMalloc((void**)&dev_Gxx,         shifted_kernel_sz *sizeof(T)) );
    cudacheck( cudaMalloc((void**)&dev_Gxxx,        shifted_kernel_sz *sizeof(T)) );
    cudacheck( cudaMalloc((void**)&dev_G_of_x,      shifted_kernel_sz *sizeof(T)) );
    cudacheck( cudaMalloc((void**)&dev_Gx_sh,       shifted_kernel_sz *sizeof(T)) );
    cudacheck( cudaMalloc((void**)&dev_Gxx_sh,      shifted_kernel_sz *sizeof(T)) );
    cudacheck( cudaMalloc((void**)&dev_Gxxx_sh,     shifted_kernel_sz *sizeof(T)) );
    cudacheck( cudaMalloc((void**)&dev_G_of_x_sh,   shifted_kernel_sz *sizeof(T)) );

    cudacheck( cudaMalloc((void**)&dev_subpix_pos_x_map,       interp_img_height*interp_img_width*sizeof(T)) );
    cudacheck( cudaMalloc((void**)&dev_subpix_pos_y_map,       interp_img_height*interp_img_width*sizeof(T)) );

    time_conv = 0;
    time_nms = 0;

    // cuda event
    cudacheck( cudaEventCreate(&start) );
    cudacheck( cudaEventCreate(&stop) );
}

// ========================= preprocessing ==========================
// Initialize 2d arrays
// ==================================================================
template<typename T>
void ThirdOrderEdgeDetectionGPU<T>::preprocessing(std::ifstream& scan_infile) {

    // -- input img initialization --
    for (int i = 0; i < img_height; i++) {
        for (int j = 0; j < img_width; j++) {
            img(i, j) = (int)scan_infile.get();
        }
    }

    // -- or, read a gray image from file directly --
    //read_array_from_file("img_matlab.txt", img, img_height, img_width);

    // -- interpolated img initialization --
    for (int i = 0; i < interp_img_height; i++) {
        for (int j = 0; j < interp_img_width; j++) {
            Ix(i, j)         = 0;
            Iy(i, j)         = 0;
            I_grad_mag(i,j)  = 0;
            I_orient(i,j)    = 0;  

            subpix_pos_x_map(i, j)       = 0;
            subpix_pos_y_map(i, j)       = 0;
        }
    }

    // TODO: push subpixel positions and orientation to the subpix_edge_pts_final list
    /*for (int i = 0; i < img_height*img_width; i++) {
        for (int j = 0; j < num_of_edge_data; j++) {
            subpix_edge_pts_final(i, j)  = 0;
        }
    }*/

    // --------------------------------------------------------------------------------------
    // gpu
    cudacheck( cudaMemset(dev_Ix,                     0, interp_img_height*interp_img_width*sizeof(T)) );
    cudacheck( cudaMemset(dev_Iy,                     0, interp_img_height*interp_img_width*sizeof(T)) );
    cudacheck( cudaMemset(dev_I_grad_mag,             0, interp_img_height*interp_img_width*sizeof(T)) );
    cudacheck( cudaMemset(dev_I_orient,               0, interp_img_height*interp_img_width*sizeof(T)) );

    cudacheck( cudaMemset(dev_subpix_pos_y_map,       0, interp_img_height*interp_img_width*sizeof(T)) );
    cudacheck( cudaMemset(dev_subpix_pos_x_map,       0, interp_img_height*interp_img_width*sizeof(T)) );
}

template<typename T>
void ThirdOrderEdgeDetectionGPU<T>::convolve_img()
{
    const int cent = (kernel_sz-1)/2;
    const int cent_interp = cent+1;

/*    // -- 2 types of filters
    T Gxxx[shifted_kernel_sz],    Gxx[shifted_kernel_sz],    Gx[shifted_kernel_sz],    G_of_x[shifted_kernel_sz];
    T Gxxx_sh[shifted_kernel_sz], Gxx_sh[shifted_kernel_sz], Gx_sh[shifted_kernel_sz], G_of_x_sh[shifted_kernel_sz];

    // -- 1D convolution filter --
    T dx = 0, dy = 0;
    for (int p = -cent_interp; p <= cent_interp; p++) {
        Gxxx[p+cent_interp]   = (((p+dy)*(3*g_sig*g_sig-(p+dy)*(p+dy)))*std::exp(-(p+dy)*(p+dy)/(2*g_sig*g_sig)))/(std::sqrt(2*PI)*g_sig*g_sig*g_sig*g_sig*g_sig*g_sig*g_sig);
        Gxx[p+cent_interp]    = (((p+dy)*(p+dy)-g_sig*g_sig)*std::exp(-(p+dy)*(p+dy)/(2*g_sig*g_sig)))/(std::sqrt(2*PI)*g_sig*g_sig*g_sig*g_sig*g_sig);
        Gx[p+cent_interp]     = (-(p+dy)*std::exp(-(p+dy)*(p+dy)/(2*g_sig*g_sig)))/(std::sqrt(2*PI)*g_sig*g_sig*g_sig);
        G_of_x[p+cent_interp] = std::exp(-(p+dx)*(p+dx)/(2*g_sig*g_sig))/(std::sqrt(2*PI)*g_sig);
    }

    dx = 0.5;
    dy = 0.5;
    for (int p = -cent_interp; p <= cent_interp; p++) {
        Gxxx_sh[p+cent_interp]   = (((p+dy)*(3*g_sig*g_sig-(p+dy)*(p+dy)))*std::exp(-(p+dy)*(p+dy)/(2*g_sig*g_sig)))/(std::sqrt(2*PI)*g_sig*g_sig*g_sig*g_sig*g_sig*g_sig*g_sig);
        Gxx_sh[p+cent_interp]    = (((p+dy)*(p+dy)-g_sig*g_sig)*std::exp(-(p+dy)*(p+dy)/(2*g_sig*g_sig)))/(std::sqrt(2*PI)*g_sig*g_sig*g_sig*g_sig*g_sig);
        Gx_sh[p+cent_interp]     = (-(p+dy)*std::exp(-(p+dy)*(p+dy)/(2*g_sig*g_sig)))/(std::sqrt(2*PI)*g_sig*g_sig*g_sig);
        G_of_x_sh[p+cent_interp] = std::exp(-(p+dx)*(p+dx)/(2*g_sig*g_sig))/(std::sqrt(2*PI)*g_sig);
    }
*/
    T Gx[] = {1.79817087452687e-05,	0.000133830225764885,	0.000763597358165040,	0.00332388630895351,	0.0109551878084803,	0.0269954832565940,	0.0485690983747094,	0.0604926811297858,	0.0440081658455374,	0,	-0.0440081658455374,	-0.0604926811297858,	-0.0485690983747094,	-0.0269954832565940,	-0.0109551878084803,	-0.00332388630895351,	-0.000763597358165040,	-0.000133830225764885,	-1.79817087452687e-05};
    T G_of_x[] = {7.99187055345274e-06,   6.69151128824427e-05,   0.000436341347522880,   0.00221592420596900,    0.00876415024678427,    0.0269954832565940, 0.0647587978329459, 0.120985362259572,  0.176032663382150,  0.199471140200716,  0.176032663382150,  0.120985362259572,  0.0647587978329459, 0.0269954832565940, 0.00876415024678427,    0.00221592420596900,    0.000436341347522880,   6.69151128824427e-05,   7.99187055345274e-06};
    T Gxx[] = {3.84608770384913e-05,	0.000250931673309160,	0.00122721003990810,	0.00443184841193801,	0.0115029471989044,	0.0202466124424455,	0.0202371243227956,	0,	-0.0330061243841531,	-0.0498677850501791,	-0.0330061243841531,	0,	0.0202371243227956,	0.0202466124424455,	0.0115029471989044,	0.00443184841193801,	0.00122721003990810,	0.000250931673309160,	3.84608770384913e-05};
    T Gxxx[] = {7.75461189639711e-05,	0.000434948233735878,	0.00176581889075666,	0.00498582946343026,	0.00890109009439027,	0.00674887081414851,	-0.00910670594525801,	-0.0302463405648929,	-0.0302556140188070,	0,	0.0302556140188070,	0.0302463405648929,	0.00910670594525801,	-0.00674887081414851,	-0.00890109009439027,	-0.00498582946343026,	-0.00176581889075666,	-0.000434948233735878,	-7.75461189639711e-05};
    
    T G_of_x_sh[] = {2.38593182706025e-05,	0.000176297841183723,	0.00101452402864988,	0.00454678125079553,	0.0158698259178337,	0.0431386594132558,	0.0913245426945110,	0.150568716077402,	0.193334058401425,	0.193334058401425,	0.150568716077402,	0.0913245426945110,	0.0431386594132558,	0.0158698259178337,	0.00454678125079553,	0.00101452402864988,	0.000176297841183723,	2.38593182706025e-05,	2.51475364429622e-06};
    T Gx_sh[] = {5.07010513250303e-05,	0.000330558452219480,	0.00164860154655606,	0.00625182421984385,	0.0178535541575629,	0.0377463269865988,	0.0570778391840694,	0.0564632685290258,	0.0241667573001781,	-0.0241667573001781,	-0.0564632685290258,	-0.0570778391840694,	-0.0377463269865988,	-0.0178535541575629,	-0.00625182421984385,	-0.00164860154655606,	-0.000330558452219480,	-5.07010513250303e-05,	-5.97253990520353e-06};
    T Gxx_sh[] = {0.000101774904498039,	0.000575722637615595,	0.00242534650599113,	0.00745956298958641,	0.0161177919477999,	0.0222433712599600,	0.0128425138164156,	-0.0164684533209659,	-0.0453126699378339,	-0.0453126699378339,	-0.0164684533209659,	0.0128425138164156,	0.0222433712599600,	0.0161177919477999,	0.00745956298958641,	0.00242534650599113,	0.000575722637615595,	0.000101774904498039,	1.35560938637843e-05};
    T Gxxx_sh[] = {0.000190921146395817,	0.000914200719419500,	0.00311688729895755,	0.00713098700075939,	0.00920573886249338,	0.000589786359165606,	-0.0205123484567749,	-0.0344073042598751,	-0.0177474623923183,	0.0177474623923183,	0.0344073042598751,	0.0205123484567749,	-0.000589786359165606,	-0.00920573886249338,	-0.00713098700075939,	-0.00311688729895755,	-0.000914200719419500,	-0.000190921146395817,	-2.92094529738860e-05};

    // -- filters --
	cudacheck( cudaMemcpy(dev_Gx,        Gx,        shifted_kernel_sz*sizeof(T), cudaMemcpyHostToDevice) );
	cudacheck( cudaMemcpy(dev_Gxx,       Gxx,       shifted_kernel_sz*sizeof(T), cudaMemcpyHostToDevice) );
	cudacheck( cudaMemcpy(dev_Gxxx,      Gxxx,      shifted_kernel_sz*sizeof(T), cudaMemcpyHostToDevice) );
    cudacheck( cudaMemcpy(dev_G_of_x,    G_of_x,    shifted_kernel_sz*sizeof(T), cudaMemcpyHostToDevice) );
	cudacheck( cudaMemcpy(dev_Gx_sh,     Gx_sh,     shifted_kernel_sz*sizeof(T), cudaMemcpyHostToDevice) );
	cudacheck( cudaMemcpy(dev_Gxx_sh,    Gxx_sh,    shifted_kernel_sz*sizeof(T), cudaMemcpyHostToDevice) );
    cudacheck( cudaMemcpy(dev_Gxxx_sh,   Gxxx_sh,   shifted_kernel_sz*sizeof(T), cudaMemcpyHostToDevice) );
	cudacheck( cudaMemcpy(dev_G_of_x_sh, G_of_x_sh, shifted_kernel_sz*sizeof(T), cudaMemcpyHostToDevice) );

    // -- original img --
    cudacheck( cudaMemcpy(dev_img,       img,       img_height*img_width*sizeof(T), cudaMemcpyHostToDevice) );

	// convolve on gpu
	cudacheck( cudaEventRecord(start) );

	gpu_convolve(device_id, img_height, img_width, interp_img_height, interp_img_width, dev_img, dev_Ix, dev_Iy, dev_I_grad_mag, dev_I_orient, dev_Gx, dev_Gxx, dev_Gxxx, dev_G_of_x, dev_Gx_sh, dev_Gxx_sh, dev_Gxxx_sh, dev_G_of_x_sh );

	cudacheck( cudaEventRecord(stop) );
	cudacheck( cudaEventSynchronize(stop) );
	cudacheck( cudaEventElapsedTime(&time_conv, start, stop) );
    printf(" ## GPU Convolution time = %8.4f ms\n", time_conv );

    // retrieve all data from GPU
    cudacheck( cudaMemcpy(Ix,            dev_Ix,          interp_img_height*interp_img_width*sizeof(T),   cudaMemcpyDeviceToHost) );
	cudacheck( cudaMemcpy(Iy,            dev_Iy,          interp_img_height*interp_img_width*sizeof(T),   cudaMemcpyDeviceToHost) );
	cudacheck( cudaMemcpy(I_grad_mag,    dev_I_grad_mag,  interp_img_height*interp_img_width*sizeof(T),   cudaMemcpyDeviceToHost) );
    cudacheck( cudaMemcpy(I_orient,      dev_I_orient,    interp_img_height*interp_img_width*sizeof(T),   cudaMemcpyDeviceToHost) );

    #if WriteDataToFile
    write_array_to_file("Ix_gpu.txt", Ix, interp_img_height, interp_img_width);
    write_array_to_file("Iy_gpu.txt", Iy, interp_img_height, interp_img_width);
    write_array_to_file("I_grad_mag_gpu.txt", I_grad_mag, interp_img_height, interp_img_width);
    write_array_to_file("I_orient_gpu.txt", I_orient, interp_img_height, interp_img_width);
    #endif
}

template<typename T>
void ThirdOrderEdgeDetectionGPU<T>::non_maximum_suppresion()
{
    T norm_dir_x, norm_dir_y;
    T slope, fp, fm;
    T coeff_A, coeff_B, coeff_C, s, s_star;
    T max_f, subpix_grad_x, subpix_grad_y;
    T candidate_edge_pt_x, candidate_edge_pt_y;
    T subpix_grad_mag;

	cudacheck( cudaEventRecord(start) );

    gpu_nms( device_id, interp_img_height, interp_img_width, dev_Ix, dev_Iy, dev_I_grad_mag, 
             dev_subpix_pos_x_map, dev_subpix_pos_y_map );

	cudacheck( cudaEventRecord(stop) );
	cudacheck( cudaEventSynchronize(stop) );
	cudacheck( cudaEventElapsedTime(&time_nms, start, stop) );

    printf(" ## GPU NMS time = %8.4f ms\n", time_nms );

    // retrieve all data from GPU
    cudacheck( cudaMemcpy(subpix_pos_x_map,            dev_subpix_pos_x_map,          interp_img_height*interp_img_width*sizeof(T),   cudaMemcpyDeviceToHost) );
	cudacheck( cudaMemcpy(subpix_pos_y_map,            dev_subpix_pos_y_map,          interp_img_height*interp_img_width*sizeof(T),   cudaMemcpyDeviceToHost) );

    #if WriteDataToFile
    write_array_to_file("subpix_pos_x_map_gpu.txt", subpix_pos_x_map, interp_img_height, interp_img_width);
    write_array_to_file("subpix_pos_y_map_gpu.txt", subpix_pos_y_map, interp_img_height, interp_img_width);
    #endif
}

// ===================================== Write data to file for debugging =======================================
// Writes a 2d dybamically allocated array to a text file for debugging
// ==============================================================================================================
template<typename T>
void ThirdOrderEdgeDetectionGPU<T>::write_array_to_file(std::string filename, T *wr_data, int first_dim, int second_dim)
{
#define wr_data(i, j) wr_data[(i) * second_dim + (j)]

    std::cout<<"writing data to a file "<<filename<<" ..."<<std::endl;
    std::string out_file_name = "./test_files/";
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
void ThirdOrderEdgeDetectionGPU<T>::read_array_from_file(std::string filename, T *rd_data, int first_dim, int second_dim)
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
ThirdOrderEdgeDetectionGPU<T>::~ThirdOrderEdgeDetectionGPU () {
    // free memory cpu
    delete[] img;
    delete[] Ix;
    delete[] Iy;
    delete[] I_grad_mag;
    delete[] I_orient;

    delete[] subpix_pos_x_map;
    delete[] subpix_pos_y_map;

    // free memory gpu
    cudacheck( cudaFree(dev_img) );
    cudacheck( cudaFree(dev_Ix) );
    cudacheck( cudaFree(dev_Iy) );
    cudacheck( cudaFree(dev_I_grad_mag) );
    cudacheck( cudaFree(dev_I_orient) );

    cudacheck( cudaFree(dev_Gx) );
    cudacheck( cudaFree(dev_Gxx) );
    cudacheck( cudaFree(dev_Gxxx) );
    cudacheck( cudaFree(dev_G_of_x) );
    cudacheck( cudaFree(dev_Gx_sh) );
    cudacheck( cudaFree(dev_Gxx_sh) );
    cudacheck( cudaFree(dev_Gxxx_sh) );
    cudacheck( cudaFree(dev_G_of_x_sh) );

    cudacheck( cudaEventDestroy(start) );
    cudacheck( cudaEventDestroy(stop) );
}

#endif    // GPU_TOED_HPP