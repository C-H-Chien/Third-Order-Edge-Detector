#ifndef GPU_KERNELS_HPP
#define GPU_KERNELS_HPP

#include<stdio.h>
#include<assert.h>
#include<cuda.h>
#include<cuda_runtime_api.h>

// cuda error check
#define cudacheck( a )  do { \
                            cudaError_t e = a; \
                            if(e != cudaSuccess) { \
                                printf("\033[1;31m"); \
                                printf("Error in %s:%d %s\n", __func__, __LINE__, cudaGetErrorString(e)); \
                                printf("\033[0m"); \
                            }\
                        } while(0)

//#ifdef __cplusplus
//extern "C" {
//#endif

// single precision convolve
void gpu_convolve(
        int device_id,
        int h, int w, int interp_h, int interp_w,
        float* dev_img, float* dev_Ix, float* dev_Iy,
        float* dev_I_grad_mag, float* dev_I_orient,
        float* dev_Gx, float* dev_Gxx,
        float* dev_Gxxx, float* dev_G_of_x,
        float* dev_Gx_sh, float* dev_Gxx_sh,
        float* dev_Gxxx_sh, float* dev_G_of_x_sh
);

// double precision convolve
void gpu_convolve(
        int device_id,
        int h, int w, int interp_h, int interp_w,
        double* dev_img, double* dev_Ix, double* dev_Iy,
        double* dev_I_grad_mag, double* dev_I_orient,
        double* dev_Gx, double* dev_Gxx,
        double* dev_Gxxx, double* dev_G_of_x,
        double* dev_Gx_sh, double* dev_Gxx_sh,
        double* dev_Gxxx_sh, double* dev_G_of_x_sh
);


void gpu_nms(
        int device_id,
        int interp_h, int interp_w,
        float* dev_Ix, float* dev_Iy, float* dev_I_grad_mag,
        float* dev_subpix_pos_x_map,
        float* dev_subpix_pos_y_map
);

// double precision NMS
void gpu_nms(
        int device_id,
        int interp_h, int interp_w,
        double* dev_Ix, double* dev_Iy, double* dev_I_grad_mag,
        double* dev_subpix_pos_x_map,
        double* dev_subpix_pos_y_map
);
/*
void gpu_hys(
        int device_id,
        int h, int w,
        int* dev_edge_label_map,
        int* dev_final_edges,
        int npasses );
*/
//#ifdef __cplusplus
//    }
//#endif

#endif // GPU_KERNELS_HPP
