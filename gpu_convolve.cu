#include <iostream>
#include"gpu_kernels.hpp"
#include"indices.hpp"

#define cond (bx == 0 && by == 0 && tx == 0 && ty == 0)
#define SIMG_LDA(PIMGX)   ((PIMGX)+0)

template<typename T, int SUBIMGX, int SUBIMGY, int THX, int THY, int SHIFTED_KS, int CENT, int SHIFTED_CENT>
__global__
void
gpu_convolve_kernel(
        int img_height, int img_width, int interp_img_height, int interp_img_width, T* dev_img,
        T* dev_Ix, T* dev_Iy, T* dev_I_grad_mag, T* dev_I_orient,
        T* dev_Gx, T* dev_Gxx, T* dev_Gxxx, T* dev_G_of_x, T* dev_Gx_sh, T* dev_Gxx_sh, T* dev_Gxxx_sh, T* dev_G_of_x_sh)
{
#define simg(i,j)     simg[(i) * img_lda + (j)]
#define timg(i,j)     timg[(i) * img_lda + (j)]

    extern __shared__ double sdata[];

    const int tid = threadIdx.x;
    const int tx  = tid / THX;
    const int ty  = tid % THX;
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;

    const int PIMGX = SUBIMGX + (2 * SHIFTED_CENT); // padded image x-size
    const int PIMGY = SUBIMGY + (2 * SHIFTED_CENT); // padded image y-size


    const int base_gtx = bx * SUBIMGX + tx;
    const int base_gty = by * SUBIMGY + ty;
    const int img_lda  = SIMG_LDA(PIMGX);

    // shared memory ptrs
    T* simg       = (T*)sdata;
    T* sGx        = simg     + (img_lda * PIMGY);
    T* sGxx       = sGx      + (SHIFTED_KS);
    T* sGxxx      = sGxx     + (SHIFTED_KS);
    T* sG_of_x    = sGxxx    + (SHIFTED_KS);
    T* sGx_sh     = sG_of_x  + (SHIFTED_KS);
    T* sGxx_sh    = sGx_sh   + (SHIFTED_KS);
    T* sGxxx_sh   = sGxx_sh  + (SHIFTED_KS);
    T* sG_of_x_sh = sGxxx_sh + (SHIFTED_KS);

    // read gaussian filter
    int i = 0, j = 0;
    //#pragma unroll
    /*for(i = 0;  i < (SHIFTED_KS)-(THX*THY); i+= (THX*THY)) {
        sGx[i + tid]        = dev_Gx[i+tid];
        sGxx[i + tid]       = dev_Gxx[i+tid];
        sGxxx[i + tid]      = dev_Gxxx[i+tid];
        sG_of_x[i + tid]    = dev_G_of_x[i+tid];
        sGx_sh[i + tid]     = dev_Gx_sh[i+tid];
        sGxx_sh[i + tid]    = dev_Gxx_sh[i+tid];
        sGxxx_sh[i + tid]   = dev_Gxxx_sh[i+tid];
        sG_of_x_sh[i + tid] = dev_G_of_x_sh[i+tid];
    }*/

    if(tid < (SHIFTED_KS)-i) {
        sGx[i + tid]        = dev_Gx[i+tid];
        sGxx[i + tid]       = dev_Gxx[i+tid];
        sGxxx[i + tid]      = dev_Gxxx[i+tid];
        sG_of_x[i + tid]    = dev_G_of_x[i+tid];
        sGx_sh[i + tid]     = dev_Gx_sh[i+tid];
        sGxx_sh[i + tid]    = dev_Gxx_sh[i+tid];
        sGxxx_sh[i + tid]   = dev_Gxxx_sh[i+tid];
        sG_of_x_sh[i + tid] = dev_G_of_x_sh[i+tid];
    }
    __syncthreads();
    // end of reading the gaussian filters

    // read sub-image into shared memory
    #pragma unroll
    for(j = 0; j < PIMGY; j+=THY) {
        #pragma unroll
        for(i = 0; i < PIMGX; i+=THX) {
            int tx_ = base_gtx + i - SHIFTED_CENT;
            int ty_ = base_gty + j - SHIFTED_CENT;
            // read only the valid area of the sub-image
            if (i+tx < PIMGX && j+ty < PIMGY)
                simg(i+tx,j+ty) = ( tx_ >= 0 && ty_ >= 0 && tx_ < img_height && ty_ < img_width ) ? dev_img[ tx_ * img_width + ty_ ] : 0.;
        }
    }
    __syncthreads();

    // convolve
    T rfx, rfy;
    T rfxx, rfyy, rfxy;
    T rfxxy, rfxyy, rfxxx, rfyyy;
    T rgrad_mag;
    T rTO_Ix, rTO_Iy;
    T rTO_grad_mag;
    T rTO_orient;

    T* timg = &simg(SHIFTED_CENT,SHIFTED_CENT);
    #pragma unroll
    for(i = 0; i < SUBIMGX; i+=THX) {
        #pragma unroll
        for(j = 0; j < SUBIMGY; j+=THY) {
            int ii = i + tx;
            int jj = j + ty;

            rfx   = 0.;
            rfy   = 0.;
            rfxx  = 0.;
            rfyy  = 0.;
            rfxy  = 0.;
            rfxxy = 0.;
            rfxyy = 0.;
            rfxxx = 0.;
            rfyyy = 0.;

            // -- 1) loop over the 17x17 filter --
            #pragma unroll
            for (int p = -CENT; p <= CENT; p++) {
                #pragma unroll
                for (int q = -CENT; q <= CENT; q++) {
                    
                    rfx += timg(ii-p, jj-q) * sGx[q+CENT+1]     * sG_of_x[p+CENT+1];      // Gx * G_of_y
                    rfy += timg(ii-p, jj-q) * sG_of_x[q+CENT+1] *     sGx[p+CENT+1];      // G_of_x * Gy

                    rfxx  += timg(ii-p, jj-q) * sGxx[q+CENT+1]    * sG_of_x[p+CENT+1];    // Gxx * G_of_y
                    rfxy  += timg(ii-p, jj-q) * sGx[q+CENT+1]     * sGx[p+CENT+1];        // Gx * Gy
                    rfyy  += timg(ii-p, jj-q) * sG_of_x[q+CENT+1] * sGxx[p+CENT+1];       // G_of_x * Gyy
                    rfxxy += timg(ii-p, jj-q) * sGxx[q+CENT+1]    * sGx[p+CENT+1];        // Gxx * Gy
                    rfxyy += timg(ii-p, jj-q) * sGx[q+CENT+1]     * sGxx[p+CENT+1];       // Gx * Gyy
                    rfxxx += timg(ii-p, jj-q) * sGxxx[q+CENT+1]   * sG_of_x[p+CENT+1];    // Gxxx * G_of_y
                    rfyyy += timg(ii-p, jj-q) * sG_of_x[q+CENT+1] * sGxxx[p+CENT+1];      // G_of_x * Gyyy
                }
            }

            rTO_Ix = rfx * (2*rfxx*rfxx + 2*rfxy*rfxy) + rfy * (2*rfxx*rfxy + 2*rfyy*rfxy) + 2*rfx*rfy*rfxxy + rfy*rfy*rfxyy + rfx*rfx*rfxxx;
            rTO_Iy = rfx * (2*rfxx*rfxy + 2*rfyy*rfxy) + rfy * (2*rfyy*rfyy + 2*rfxy*rfxy) + 2*rfx*rfy*rfxyy + rfx*rfx*rfxxy + rfy*rfy*rfyyy;
            rTO_grad_mag = std::sqrt( rTO_Ix *rTO_Ix + rTO_Iy * rTO_Iy );
            rTO_Ix /= rTO_grad_mag;
            rTO_Iy /= rTO_grad_mag;
            rTO_orient = atan2(rTO_Ix, -rTO_Iy);

            rgrad_mag = sqrt(rfx*rfx + rfy*rfy);
            int tx_ = base_gtx + i;
            int ty_ = base_gty + j;

            if( tx_ < img_height && ty_ <  img_width) {
                dev_Ix(2*tx_,2*ty_) = rfx;
                dev_Iy(2*tx_,2*ty_) = rfy;
                dev_I_grad_mag(2*tx_,2*ty_) = rgrad_mag;
                dev_I_orient(2*tx_,2*ty_) = rTO_orient;
            }

            // --------------------------------------------------------------------------------------------------
            rfx   = 0.;
            rfy   = 0.;
            rfxx  = 0.;
            rfyy  = 0.;
            rfxy  = 0.;
            rfxxy = 0.;
            rfxyy = 0.;
            rfxxx = 0.;
            rfyyy = 0.;

            // -- 2) loop over the 19x19 filter, right top, shifted in x only --
            #pragma unroll
            for (int p = -SHIFTED_CENT; p <= SHIFTED_CENT; p++) {
                #pragma unroll
                for (int q = -SHIFTED_CENT; q <= SHIFTED_CENT; q++) {

                    rfx += timg(ii-p, jj-q) * sGx_sh[q+SHIFTED_CENT]     * sG_of_x[p+SHIFTED_CENT];
                    rfy += timg(ii-p, jj-q) * sG_of_x_sh[q+SHIFTED_CENT] *     sGx[p+SHIFTED_CENT];

                    rfxx  += timg(ii-p, jj-q) * sGxx_sh[q+SHIFTED_CENT]    * sG_of_x[p+SHIFTED_CENT];
                    rfxy  += timg(ii-p, jj-q) * sGx_sh[q+SHIFTED_CENT]     * sGx[p+SHIFTED_CENT];
                    rfyy  += timg(ii-p, jj-q) * sG_of_x_sh[q+SHIFTED_CENT] * sGxx[p+SHIFTED_CENT];
                    rfxxy += timg(ii-p, jj-q) * sGxx_sh[q+SHIFTED_CENT]    * sGx[p+SHIFTED_CENT];
                    rfxyy += timg(ii-p, jj-q) * sGx_sh[q+SHIFTED_CENT]     * sGxx[p+SHIFTED_CENT];
                    rfxxx += timg(ii-p, jj-q) * sGxxx_sh[q+SHIFTED_CENT]   * sG_of_x[p+SHIFTED_CENT];
                    rfyyy += timg(ii-p, jj-q) * sG_of_x_sh[q+SHIFTED_CENT] * sGxxx[p+SHIFTED_CENT];
                }
            }

            rTO_Ix = rfx * (2*rfxx*rfxx + 2*rfxy*rfxy) + rfy * (2*rfxx*rfxy + 2*rfyy*rfxy) + 2*rfx*rfy*rfxxy + rfy*rfy*rfxyy + rfx*rfx*rfxxx;
            rTO_Iy = rfx * (2*rfxx*rfxy + 2*rfyy*rfxy) + rfy * (2*rfyy*rfyy + 2*rfxy*rfxy) + 2*rfx*rfy*rfxyy + rfx*rfx*rfxxy + rfy*rfy*rfyyy;
            rTO_grad_mag = std::sqrt( rTO_Ix *rTO_Ix + rTO_Iy * rTO_Iy );
            rTO_Ix /= rTO_grad_mag;
            rTO_Iy /= rTO_grad_mag;
            rTO_orient = atan2(rTO_Ix, -rTO_Iy);
            rgrad_mag = sqrt(rfx*rfx + rfy*rfy);

            if( tx_ < img_height && ty_ <  img_width) {
                dev_Ix(2*tx_,2*ty_+1) = rfx;
                dev_Iy(2*tx_,2*ty_+1) = rfy;
                dev_I_grad_mag(2*tx_,2*ty_+1) = rgrad_mag;
                dev_I_orient(2*tx_,2*ty_+1) = rTO_orient;
            }
            // ----------------------------------------------------------------
            rfx   = 0.;
            rfy   = 0.;
            rfxx  = 0.;
            rfyy  = 0.;
            rfxy  = 0.;
            rfxxy = 0.;
            rfxyy = 0.;
            rfxxx = 0.;
            rfyyy = 0.;

            // -- 3) loop over the 19x19 filter, left bottom, shifted in y only --
            #pragma unroll
            for (int p = -SHIFTED_CENT; p <= SHIFTED_CENT; p++) {
                #pragma unroll
                for (int q = -SHIFTED_CENT; q <= SHIFTED_CENT; q++) {
                    rfx += timg(ii-p, jj-q) * sGx[q+SHIFTED_CENT]     * sG_of_x_sh[p+SHIFTED_CENT];      // Gx * G_of_y
                    rfy += timg(ii-p, jj-q) * sG_of_x[q+SHIFTED_CENT] *     sGx_sh[p+SHIFTED_CENT];      // G_of_x * Gy

                    rfxx  += timg(ii-p, jj-q) * sGxx[q+SHIFTED_CENT]    * sG_of_x_sh[p+SHIFTED_CENT];    // Gxx * G_of_y
                    rfxy  += timg(ii-p, jj-q) * sGx[q+SHIFTED_CENT]     * sGx_sh[p+SHIFTED_CENT];        // Gx * Gy
                    rfyy  += timg(ii-p, jj-q) * sG_of_x[q+SHIFTED_CENT] * sGxx_sh[p+SHIFTED_CENT];       // G_of_x * Gyy
                    rfxxy += timg(ii-p, jj-q) * sGxx[q+SHIFTED_CENT]    * sGx_sh[p+SHIFTED_CENT];        // Gxx * Gy
                    rfxyy += timg(ii-p, jj-q) * sGx[q+SHIFTED_CENT]     * sGxx_sh[p+SHIFTED_CENT];       // Gx * Gyy
                    rfxxx += timg(ii-p, jj-q) * sGxxx[q+SHIFTED_CENT]   * sG_of_x_sh[p+SHIFTED_CENT];    // Gxxx * G_of_y
                    rfyyy += timg(ii-p, jj-q) * sG_of_x[q+SHIFTED_CENT] * sGxxx_sh[p+SHIFTED_CENT];      // G_of_x * Gyyy
                }
            }

            rTO_Ix = rfx * (2*rfxx*rfxx + 2*rfxy*rfxy) + rfy * (2*rfxx*rfxy + 2*rfyy*rfxy) + 2*rfx*rfy*rfxxy + rfy*rfy*rfxyy + rfx*rfx*rfxxx;
            rTO_Iy = rfx * (2*rfxx*rfxy + 2*rfyy*rfxy) + rfy * (2*rfyy*rfyy + 2*rfxy*rfxy) + 2*rfx*rfy*rfxyy + rfx*rfx*rfxxy + rfy*rfy*rfyyy;
            rTO_grad_mag = std::sqrt( rTO_Ix *rTO_Ix + rTO_Iy * rTO_Iy );
            rTO_Ix /= rTO_grad_mag;
            rTO_Iy /= rTO_grad_mag;
            rTO_orient = atan2(rTO_Ix, -rTO_Iy);
            rgrad_mag = sqrt(rfx*rfx + rfy*rfy);

            if( tx_ < img_height && ty_ <  img_width) {
                dev_Ix(2*tx_+1,2*ty_) = rfx;
                dev_Iy(2*tx_+1,2*ty_) = rfy;
                dev_I_grad_mag(2*tx_+1,2*ty_) = rgrad_mag;
                dev_I_orient(2*tx_+1,2*ty_) = rTO_orient;
            }
            //----------------------------------------------------------------
            rfx   = 0.;
            rfy   = 0.;
            rfxx  = 0.;
            rfyy  = 0.;
            rfxy  = 0.;
            rfxxy = 0.;
            rfxyy = 0.;
            rfxxx = 0.;
            rfyyy = 0.;

            // -- 4) loop over the 19x19 filter, right bottom, shifted in both x and y --
            #pragma unroll
            for (int p = -SHIFTED_CENT; p <= SHIFTED_CENT; p++) {
                #pragma unroll
                for (int q = -SHIFTED_CENT; q <= SHIFTED_CENT; q++) {
                    rfx += timg(ii-p, jj-q) * sGx_sh[q+SHIFTED_CENT]     * sG_of_x_sh[p+SHIFTED_CENT];      // Gx * G_of_y
                    rfy += timg(ii-p, jj-q) * sG_of_x_sh[q+SHIFTED_CENT] *     sGx_sh[p+SHIFTED_CENT];      // G_of_x * Gy

                    rfxx  += timg(ii-p, jj-q) * sGxx_sh[q+SHIFTED_CENT]    * sG_of_x_sh[p+SHIFTED_CENT];    // Gxx * G_of_y
                    rfxy  += timg(ii-p, jj-q) * sGx_sh[q+SHIFTED_CENT]     * sGx_sh[p+SHIFTED_CENT];        // Gx * Gy
                    rfyy  += timg(ii-p, jj-q) * sG_of_x_sh[q+SHIFTED_CENT] * sGxx_sh[p+SHIFTED_CENT];       // G_of_x * Gyy
                    rfxxy += timg(ii-p, jj-q) * sGxx_sh[q+SHIFTED_CENT]    * sGx_sh[p+SHIFTED_CENT];        // Gxx * Gy
                    rfxyy += timg(ii-p, jj-q) * sGx_sh[q+SHIFTED_CENT]     * sGxx_sh[p+SHIFTED_CENT];       // Gx * Gyy
                    rfxxx += timg(ii-p, jj-q) * sGxxx_sh[q+SHIFTED_CENT]   * sG_of_x_sh[p+SHIFTED_CENT];    // Gxxx * G_of_y
                    rfyyy += timg(ii-p, jj-q) * sG_of_x_sh[q+SHIFTED_CENT] * sGxxx_sh[p+SHIFTED_CENT];      // G_of_x * Gyyy
                }
            }

            rTO_Ix = rfx * (2*rfxx*rfxx + 2*rfxy*rfxy) + rfy * (2*rfxx*rfxy + 2*rfyy*rfxy) + 2*rfx*rfy*rfxxy + rfy*rfy*rfxyy + rfx*rfx*rfxxx;
            rTO_Iy = rfx * (2*rfxx*rfxy + 2*rfyy*rfxy) + rfy * (2*rfyy*rfyy + 2*rfxy*rfxy) + 2*rfx*rfy*rfxyy + rfx*rfx*rfxxy + rfy*rfy*rfyyy;
            rTO_grad_mag = std::sqrt( rTO_Ix *rTO_Ix + rTO_Iy * rTO_Iy );
            rTO_Ix /= rTO_grad_mag;
            rTO_Iy /= rTO_grad_mag;
            rTO_orient = atan2(rTO_Ix, -rTO_Iy);
            rgrad_mag = sqrt(rfx*rfx + rfy*rfy);

            if( tx_ < img_height && ty_ <  img_width) {
                dev_Ix(2*tx_+1,2*ty_+1) = rfx;
                dev_Iy(2*tx_+1,2*ty_+1) = rfy;
                dev_I_grad_mag(2*tx_+1,2*ty_+1) = rgrad_mag;
                dev_I_orient(2*tx_+1,2*ty_+1) = rTO_orient;
            }
        }
    }

    // TODO: write back in a separate loop
}


//extern "C"
template<typename T>
void gpu_convolve_template(
        int device_id,
        int h, int w, int interp_h, int interp_w,
        T* dev_img, T* dev_Ix, T* dev_Iy,
        T* dev_I_grad_mag, T* dev_I_orient,
        T* dev_Gx, T* dev_Gxx,
        T* dev_Gxxx, T* dev_G_of_x,
        T* dev_Gx_sh, T* dev_Gxx_sh,
        T* dev_Gxxx_sh, T* dev_G_of_x_sh )
{
    // kernel parameters
    // TODO: tune
    const int ks   = 17;
    const int cent = (ks-1)/2;
    const int shifted_ks = 19;
    const int shift_cent = cent + 1;
    const int subimgx = 8;
    const int subimgy = 8;
    const int thx     = 8;
    const int thy     = 8;
    // end of kernel parameters

    const int gridx = (h + subimgx - 1) / subimgx;
    const int gridy = (w + subimgy - 1) / subimgy;
    dim3 grid(gridx, gridy, 1);
    dim3 threads(thx*thy,1,1);

    const int simg_lda = SIMG_LDA(subimgx + shift_cent + shift_cent);

    int shmem = 0;
    shmem += sizeof(T) * simg_lda * (subimgx + shift_cent + shift_cent);  // padded sub-image
    shmem += sizeof(T) * (shifted_ks); // Gx
    shmem += sizeof(T) * (shifted_ks); // Gxx
    shmem += sizeof(T) * (shifted_ks); // Gxxx
    shmem += sizeof(T) * (shifted_ks); // G_of_x
    shmem += sizeof(T) * (shifted_ks); // Gx_sh
    shmem += sizeof(T) * (shifted_ks); // Gxx_sh
    shmem += sizeof(T) * (shifted_ks); // Gxxx_sh
    shmem += sizeof(T) * (shifted_ks); // G_of_x_sh

    // get max. dynamic shared memory on the GPU
    int nthreads_max, shmem_max = 0;
    cudacheck( cudaDeviceGetAttribute(&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device_id) );
    #if CUDA_VERSION >= 9000
    cudacheck( cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id) );
    if (shmem <= shmem_max) {
        cudacheck( cudaFuncSetAttribute(gpu_convolve_kernel<T, subimgx, subimgy, thx, thy, shifted_ks, cent, shift_cent>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem) );
    }
    #else
    cudacheck( cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device_id) );
    #endif    // CUDA_VERSION >= 9000

    if ( shmem > shmem_max ) {
        printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
    }

    void *kernel_args[] = {&h, &w, &interp_h, &interp_w, &dev_img, &dev_Ix, &dev_Iy, &dev_I_grad_mag, &dev_I_orient, &dev_Gx, &dev_Gxx, &dev_Gxxx, &dev_G_of_x, &dev_Gx_sh, &dev_Gxx_sh, &dev_Gxxx_sh, &dev_G_of_x_sh};

    cudacheck( cudaLaunchKernel((void*)gpu_convolve_kernel<T, subimgx, subimgy, thx, thy, shifted_ks, cent, shift_cent>, grid, threads, kernel_args, shmem, NULL) );
}


// single precision
void gpu_convolve(
        int device_id,
        int h, int w, int interp_h, int interp_w,
        float* dev_img, float* dev_Ix, float* dev_Iy,
        float* dev_I_grad_mag, float* dev_I_orient,
        float* dev_Gx, float* dev_Gxx,
        float* dev_Gxxx, float* dev_G_of_x,
        float* dev_Gx_sh, float* dev_Gxx_sh,
        float* dev_Gxxx_sh, float* dev_G_of_x_sh)
{
    gpu_convolve_template<float>(device_id, h, w, interp_h, interp_w,
                                 dev_img, dev_Ix, dev_Iy, dev_I_grad_mag, dev_I_orient,
                                 dev_Gx, dev_Gxx, dev_Gxxx, dev_G_of_x,
                                 dev_Gx_sh, dev_Gxx_sh, dev_Gxxx_sh, dev_G_of_x_sh);
}


// double precision
void gpu_convolve(
        int device_id,
        int h, int w, int interp_h, int interp_w,
        double* dev_img, double* dev_Ix, double* dev_Iy,
        double* dev_I_grad_mag, double* dev_I_orient,
        double* dev_Gx, double* dev_Gxx,
        double* dev_Gxxx, double* dev_G_of_x,
        double* dev_Gx_sh, double* dev_Gxx_sh,
        double* dev_Gxxx_sh, double* dev_G_of_x_sh)
{
    gpu_convolve_template<double>(device_id, h, w, interp_h, interp_w,
                                  dev_img, dev_Ix, dev_Iy, dev_I_grad_mag, dev_I_orient,
                                  dev_Gx, dev_Gxx, dev_Gxxx, dev_G_of_x,
                                  dev_Gx_sh, dev_Gxx_sh, dev_Gxxx_sh, dev_G_of_x_sh);
}
