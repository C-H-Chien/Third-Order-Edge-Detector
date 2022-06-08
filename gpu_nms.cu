#include"gpu_kernels.hpp"
#include"indices.hpp"

#define cond (bx == 14 && by == 14 && tx == 0 && ty == 0)

#define SQRT2 (1.4142135623731)

template<typename T, int SUBIMGX, int SUBIMGY, int THX, int THY, int PADDING, int SN>
__global__
void
gpu_nms_kernel(
        int interp_img_height, int interp_img_width,
        T* dev_Ix, T* dev_Iy, T* dev_I_grad_mag,
        T* dev_subpix_pos_x_map,
        T* dev_subpix_pos_y_map )
{
#define sIx(i,j)             sIx[(i) * PIMGX + (j)]
#define sIy(i,j)             sIy[(i) * PIMGX + (j)]
#define sgrad_mag(i,j) sgrad_mag[(i) * PIMGX + (j)]

#define tIx(i,j)             tIx[(i) * PIMGX + (j)]
#define tIy(i,j)             tIy[(i) * PIMGX + (j)]
#define tgrad_mag(i,j) tgrad_mag[(i) * PIMGX + (j)]

    extern __shared__ double sdata[];

    const int tid = threadIdx.x;
    const int tx  = tid / THX;
    const int ty  = tid % THX;
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;

    const int PIMGX = SUBIMGX + (2 * PADDING); // padded image x-size
    const int PIMGY = SUBIMGY + (2 * PADDING); // padded image y-size

    const int base_gtx = bx * SUBIMGX + tx + PADDING;
    const int base_gty = by * SUBIMGY + ty + PADDING;

    // shared memory ptrs
    T* sIx       = (T*)sdata;
    T* sIy       = sIx + PIMGX * PIMGY;
    T* sgrad_mag = sIy + PIMGX * PIMGY;

    T* tIx       = &sIx(PADDING, PADDING);
    T* tIy       = &sIy(PADDING, PADDING);
    T* tgrad_mag = &sgrad_mag(PADDING, PADDING);

    // read sIx, sIy, and sgrad_mag
    int ii = 0, jj = 0;
    #pragma unroll
    for(jj = 0; jj < PIMGY-THY; jj+=THY) {
        int ty_ = base_gty + jj - PADDING;
        #pragma unroll
        for(ii = 0; ii < PIMGX-THX; ii+=THX) {
            int tx_ = base_gtx + ii - PADDING;
            sIx(ii+tx,jj+ty)       = (tx_ < interp_img_height && ty_ < interp_img_width ) ? dev_Ix(tx_, ty_) : 0.;
            sIy(ii+tx,jj+ty)       = (tx_ < interp_img_height && ty_ < interp_img_width ) ? dev_Iy(tx_, ty_) : 0.;
            sgrad_mag(ii+tx,jj+ty) = (tx_ < interp_img_height && ty_ < interp_img_width ) ? dev_I_grad_mag(tx_, ty_) : 0.;
        }

        if(tx < PIMGX-ii) {
            int tx_ = base_gtx + ii - PADDING;
            sIx(ii+tx,jj+ty)       = (tx_ < interp_img_height && ty_ < interp_img_width ) ? dev_Ix(tx_, ty_) : 0.;
            sIy(ii+tx,jj+ty)       = (tx_ < interp_img_height && ty_ < interp_img_width ) ? dev_Iy(tx_, ty_) : 0.;
            sgrad_mag(ii+tx,jj+ty) = (tx_ < interp_img_height && ty_ < interp_img_width ) ? dev_I_grad_mag(tx_, ty_) : 0.;
        }
    }

    // last column block
    if(ty < PIMGY-jj) {
        int ty_ = base_gty + jj - PADDING;
        #pragma unroll
        for(ii = 0; ii < PIMGX-THX; ii+=THX) {
            int tx_ = base_gtx + ii - PADDING;
            sIx(ii+tx,jj+ty)       = (tx_ < interp_img_height && ty_ < interp_img_width ) ? dev_Ix(tx_, ty_) : 0.;
            sIy(ii+tx,jj+ty)       = (tx_ < interp_img_height && ty_ < interp_img_width ) ? dev_Iy(tx_, ty_) : 0.;
            sgrad_mag(ii+tx,jj+ty) = (tx_ < interp_img_height && ty_ < interp_img_width ) ? dev_I_grad_mag(tx_, ty_) : 0.;
        }

        if(tx < PIMGX-ii) {
            int tx_ = base_gtx + ii - PADDING;
            sIx(ii+tx,jj+ty)       = (tx_ < interp_img_height && ty_ < interp_img_width ) ? dev_Ix(tx_, ty_) : 0.;
            sIy(ii+tx,jj+ty)       = (tx_ < interp_img_height && ty_ < interp_img_width ) ? dev_Iy(tx_, ty_) : 0.;
            sgrad_mag(ii+tx,jj+ty) = (tx_ < interp_img_height && ty_ < interp_img_width ) ? dev_I_grad_mag(tx_, ty_) : 0.;
        }
    }
    __syncthreads();

    int i = tx, j = ty;
    T norm_dir_x, norm_dir_y;
    T slope, fp, fm;
    T coeff_A, coeff_B, coeff_C, s, s_star;
    T max_f, subpix_grad_x, subpix_grad_y;
    T subpix_pos_x_map = 0, subpix_pos_y_map = 0;

    // -- ignore neglectable gradient magnitude --
    if (tgrad_mag(i, j) <= 2) return;

    // -- ignore invalid gradient direction --
    if ( (fabs(tIx(i, j)) < 1e-6) && (fabs(tIy(i, j)) < 1e-6) ) return;

    // -- calculate the unit direction --
    norm_dir_x = tIx(i,j) / tgrad_mag(i,j);
    norm_dir_y = tIy(i,j) / tgrad_mag(i,j);

    // -- find corresponding quadrant --
    if ((tIx(i,j) >= 0) && (tIy(i,j) >= 0)) {
        if (tIx(i,j) >= tIy(i,j)) {         // -- 1st quadrant --
            slope = norm_dir_y / norm_dir_x;
            fp = tgrad_mag(i, j+SN) * (1-slope) + tgrad_mag(i+SN, j+SN) * slope;
            fm = tgrad_mag(i, j-SN) * (1-slope) + tgrad_mag(i-SN, j-SN) * slope;
        }
        else {                              // -- 2nd quadrant --
            slope = norm_dir_x / norm_dir_y;
            fp = tgrad_mag(i+SN, j) * (1-slope) + tgrad_mag(i+SN, j+SN) * slope;
            fm = tgrad_mag(i-SN, j) * (1-slope) + tgrad_mag(i-SN, j-SN) * slope;
        }
    }
    else if ((tIx(i,j) < 0) && (tIy(i,j) >= 0)) {
        if (fabs(tIx(i,j)) < tIy(i,j)) {     // -- 3rd quadrant --
            slope = -norm_dir_x / norm_dir_y;
            fp = tgrad_mag(i+SN, j) * (1-slope) + tgrad_mag(i+SN, j-SN) * slope;
            fm = tgrad_mag(i-SN, j) * (1-slope) + tgrad_mag(i-SN, j+SN)  * slope;
        }
        else {                              // -- 4th quadrant --
            slope = -norm_dir_y / norm_dir_x;
            fp = tgrad_mag(i, j-SN) * (1-slope) + tgrad_mag(i+SN, j-SN) * slope;
            fm = tgrad_mag(i, j+SN) * (1-slope) + tgrad_mag(i-SN, j+SN) * slope;
        }
    }
    else if ((tIx(i,j) < 0) && (tIy(i,j) < 0)) {
        if(fabs(tIx(i,j)) >= fabs(tIy(i,j))) {            // -- 5th quadrant --
            slope = norm_dir_y / norm_dir_x;
            fp = tgrad_mag(i, j-SN) * (1-slope) + tgrad_mag(i-SN, j-SN) * slope;
            fm = tgrad_mag(i, j+SN) * (1-slope) + tgrad_mag(i+SN, j+SN) * slope;
        }
        else {                              // -- 6th quadrant --
            slope = norm_dir_x / norm_dir_y;
            fp = tgrad_mag(i-SN, j) * (1-slope) + tgrad_mag(i-SN, j-SN) * slope;
            fm = tgrad_mag(i+SN, j) * (1-slope) + tgrad_mag(i+SN, j+SN) * slope;
        }
    }
    else if ((tIx(i,j) >= 0) && (tIy(i,j) < 0)) {
        if(tIx(i,j) < fabs(tIy(i,j))) {      // -- 7th quadrant --
            slope = -norm_dir_x / norm_dir_y;
            fp = tgrad_mag(i-SN, j) * (1-slope) + tgrad_mag(i-SN, j+SN) * slope;
            fm = tgrad_mag(i+SN, j) * (1-slope) + tgrad_mag(i+SN, j-SN) * slope;
        }
        else {                              // -- 8th quadrant --
            slope = -norm_dir_y / norm_dir_x;
            fp = tgrad_mag(i, j+SN) * (1-slope) + tgrad_mag(i-SN, j+SN) * slope;
            fm = tgrad_mag(i, j-SN) * (1-slope) + tgrad_mag(i+SN, j-SN) * slope;
        }
    }

    // -- fit a parabola to find the edge subpixel location when doing max test --
    s = sqrt(1+slope*slope);
    if((tgrad_mag(i, j) >  fm && tgrad_mag(i, j) > fp) ||  // -- abs max --
       (tgrad_mag(i, j) >  fm && tgrad_mag(i, j) >= fp) || // -- relaxed max --
       (tgrad_mag(i, j) >= fm && tgrad_mag(i, j) >  fp))
    {

        // -- fit a parabola; define coefficients --
        coeff_A = (fm+fp-2*tgrad_mag(i, j))/(2*s*s);
        coeff_B = (fp-fm)/(2*s);
        coeff_C = tgrad_mag(i, j);

        s_star = -coeff_B/(2*coeff_A); // -- location of max --
        max_f = coeff_A*s_star*s_star + coeff_B*s_star + coeff_C; // -- value of max --

        if(fabs(s_star) <= SQRT2) { // -- significant max is within a pixel --

            // -- subpixel magnitude in x and y --
            //subpix_grad_x = max_f*norm_dir_x;
            //subpix_grad_y = max_f*norm_dir_y;

            // -- subpixel gradient magnitude --
            //subpix_grad_mag = sqrt(subpix_grad_x*subpix_grad_x + subpix_grad_y*subpix_grad_y);

            // -- store subpixel location in a map --
            subpix_pos_x_map = base_gty/*j*/ + s_star * norm_dir_x;
            subpix_pos_y_map = base_gtx/*i*/ + s_star * norm_dir_y;

            // -- store gradient of subpixel edge in the map --
            //subpix_grad_mag_map = subpix_grad_mag;
        }
    }
    __syncthreads();

    if(base_gtx < interp_img_height-PADDING && base_gty < interp_img_width-PADDING) {
        dev_subpix_pos_x_map(base_gtx, base_gty)       = subpix_pos_x_map;
        dev_subpix_pos_y_map(base_gtx, base_gty)       = subpix_pos_y_map;
    }
}


//extern "C"
template<typename T>
void gpu_nms_template(
        int device_id,
        int interp_h, int interp_w,
        T* dev_Ix, T* dev_Iy, T* dev_I_grad_mag,
        T* dev_subpix_pos_x_map,
        T* dev_subpix_pos_y_map )
{
    // kernel parameters
    // TODO: tune
    const int padding = 10;
    const int subimgx = 8;
    const int subimgy = 8;
    const int thx     = 8;
    const int thy     = 8;
    const int sn      = 1; // should be 1 but now just use 2
    // end of kernel parameters

    assert(subimgx == thx && subimgy == thy && thx == thy);

    int hh = interp_h - (2*padding);
    int ww = interp_w - (2*padding);

    const int gridx = (hh + subimgx - 1) / subimgx;
    const int gridy = (ww + subimgy - 1) / subimgy;
    dim3 grid(gridx, gridy, 1);
    dim3 threads(thx*thy,1,1);

    int shmem = 0;
    shmem += sizeof(T) * (subimgx + padding + padding) * (subimgy + padding + padding);  // sIx
    shmem += sizeof(T) * (subimgx + padding + padding) * (subimgy + padding + padding);  // sIy
    shmem += sizeof(T) * (subimgx + padding + padding) * (subimgy + padding + padding);  // sgrad_mag

    // get max. dynamic shared memory on the GPU
    int nthreads_max, shmem_max = 0;
    cudacheck( cudaDeviceGetAttribute(&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device_id) );
    #if CUDA_VERSION >= 9000
    cudacheck( cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id) );
    if (shmem <= shmem_max) {
        cudacheck( cudaFuncSetAttribute(gpu_nms_kernel<T, subimgx, subimgy, thx, thy, padding, sn>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem) );
    }
    #else
    cudacheck( cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device_id) );
    #endif    // CUDA_VERSION >= 9000

    if ( shmem > shmem_max ) {
        printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
    }

    void *kernel_args[] = { &interp_h, &interp_w, 
                            &dev_Ix, &dev_Iy, &dev_I_grad_mag,
                            &dev_subpix_pos_x_map, &dev_subpix_pos_y_map };

    cudacheck( cudaLaunchKernel((void*)gpu_nms_kernel<T, subimgx, subimgy, thx, thy, padding, sn>, grid, threads, kernel_args, shmem, NULL) );
}


// single precision NMS
void gpu_nms(
        int device_id,
        int interp_h, int interp_w,
        float* dev_Ix, float* dev_Iy, float* dev_I_grad_mag,
        float* dev_subpix_pos_x_map,
        float* dev_subpix_pos_y_map )
{
    gpu_nms_template<float>(
        device_id, interp_h, interp_w,
        dev_Ix, dev_Iy, dev_I_grad_mag,
        dev_subpix_pos_x_map, dev_subpix_pos_y_map
    );
}

// double precision NMS
void gpu_nms(
        int device_id,
        int interp_h, int interp_w,
        double* dev_Ix, double* dev_Iy, double* dev_I_grad_mag,
        double* dev_subpix_pos_x_map,
        double* dev_subpix_pos_y_map )
{
    gpu_nms_template<double>(
        device_id, interp_h, interp_w,
        dev_Ix, dev_Iy, dev_I_grad_mag,
        dev_subpix_pos_x_map, dev_subpix_pos_y_map
    );
}
