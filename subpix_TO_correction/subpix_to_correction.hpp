#ifndef SUBPIX_TO_CORRECTION_HPP
#define SUBPIX_TO_CORRECTION_HPP

#include <fstream>
#include <string>
#include <vector>

#include "indices.hpp"

#if OPENCV_SUPPORT
#include <opencv2/opencv.hpp>
#endif

// Subpixel third-order correction on a dense edge map.
// Port of MATLAB subpix_TO_correction.m (called from edgesDetect_TO.m):
//   NMS along the given orientation, then third-order orientation
//   correction from Gaussian derivatives of the edge map.
// Output edgels are [x, y, orientation, strength] in 0-based pixel coords
// (same convention as TOED_edges.txt / data_final_output_cpu.txt).
template<typename T>
class SubpixelTOCorrectionCPU {
    int img_height;
    int img_width;
    int work_height;
    int work_width;
    int interp_n;
    int omp_threads;
    int kernel_rad;
    T thresh;
    T sigma;
    T neighbor_radius;

    T *edgemap;
    T *theta;
    T *dirx;
    T *diry;
    T *Hx;
    T *Hy;
    T *Px;
    T *Py;
    T *Pxx;
    T *Pxy;
    T *Pyy;
    T *Hxx;
    T *Hxy;
    T *Hyy;
    T *conv_tmp;

    T *nms_x;
    T *nms_y;
    T *nms_mag;
    unsigned char *nms_valid;

    T *G_1d;
    T *dG_1d;
    T *d2G_1d;

    std::string output_dir;

    void allocate_work_buffers(int h, int w);
    void free_work_buffers();
    void build_gaussian_kernels();
    void compute_dir_from_theta();
    void conv_separable_circular(const T *src, T *dst, const T *kx, const T *ky);
    void nms_token(int margin);
    T interp2_cubic(const T *src, T x, T y) const;
    T wrap_mod_pi(T a) const;

  public:
    T *subpix_edge_pts_final;
    T *TO_edgemap;
    T *TO_orientation;
    int edge_pt_list_idx;
    int num_of_edge_data;

    double time_nms;
    double time_conv;
    double time_correct;

    SubpixelTOCorrectionCPU(int H, int W, T threshold, T g_sigma, int cpu_nthreads, int n = 0);
    ~SubpixelTOCorrectionCPU();

    void set_edgemap(const T *data);
    void set_orientation(const T *data);
    void estimate_orientation();

#if OPENCV_SUPPORT
    void set_edgemap(const cv::Mat &image);
    void set_orientation(const cv::Mat &image);
#endif

    int run(T *edginfo_out);
    void set_output_dir(const std::string &dir);
    void set_neighbor_radius(T radius);
    void write_array_to_file(std::string filename, T *wr_data, int first_dim, int second_dim);
};

#endif
