#ifndef SUBPIX_TO_CORRECTION_CPP
#define SUBPIX_TO_CORRECTION_CPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

#include <omp.h>

#include "subpix_to_correction.hpp"

#define Emap(i, j)           edgemap[(i) * work_width + (j)]
#define Theta(i, j)          theta[(i) * work_width + (j)]
#define Dirx(i, j)           dirx[(i) * work_width + (j)]
#define Diry(i, j)           diry[(i) * work_width + (j)]
#define Hx_(i, j)            Hx[(i) * work_width + (j)]
#define Hy_(i, j)            Hy[(i) * work_width + (j)]
#define Px_(i, j)            Px[(i) * work_width + (j)]
#define Py_(i, j)            Py[(i) * work_width + (j)]
#define Pxx_(i, j)           Pxx[(i) * work_width + (j)]
#define Pxy_(i, j)           Pxy[(i) * work_width + (j)]
#define Pyy_(i, j)           Pyy[(i) * work_width + (j)]
#define Hxx_(i, j)           Hxx[(i) * work_width + (j)]
#define Hxy_(i, j)           Hxy[(i) * work_width + (j)]
#define Hyy_(i, j)           Hyy[(i) * work_width + (j)]
#define nms_x_(i, j)         nms_x[(i) * work_width + (j)]
#define nms_y_(i, j)         nms_y[(i) * work_width + (j)]
#define nms_mag_(i, j)       nms_mag[(i) * work_width + (j)]
#define nms_valid_(i, j)     nms_valid[(i) * work_width + (j)]
#define TO_E(i, j)           TO_edgemap[(i) * img_width + (j)]
#define TO_O(i, j)           TO_orientation[(i) * img_width + (j)]
#define edg(i, j)            subpix_edge_pts_final[(i) * num_of_edge_data + (j)]
#define out_e(i, j)          edginfo_out[(i) * num_of_edge_data + (j)]

static inline int wrap_idx(int i, int n)
{
    int r = i % n;
    if (r < 0) r += n;
    return r;
}

static inline int wrap_symmetric(int i, int n)
{
    if (n <= 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i - 1;
        else       i = 2 * n - i - 1;
    }
    return i;
}

template<typename T>
static inline T cubic_kernel(T s)
{
    const T a = static_cast<T>(-0.5);
    T x = std::abs(s);
    if (x <= static_cast<T>(1))
        return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < static_cast<T>(2))
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
    return 0;
}

template<typename T>
static void bilinear_resize(const T *src, int sh, int sw, T *dst, int dh, int dw)
{
    for (int i = 0; i < dh; i++) {
        T y = (dh == 1) ? 0 : static_cast<T>(i) * (sh - 1) / static_cast<T>(dh - 1);
        int y0 = static_cast<int>(std::floor(y));
        int y1 = std::min(y0 + 1, sh - 1);
        T fy = y - y0;
        for (int j = 0; j < dw; j++) {
            T x = (dw == 1) ? 0 : static_cast<T>(j) * (sw - 1) / static_cast<T>(dw - 1);
            int x0 = static_cast<int>(std::floor(x));
            int x1 = std::min(x0 + 1, sw - 1);
            T fx = x - x0;
            T v00 = src[y0 * sw + x0];
            T v01 = src[y0 * sw + x1];
            T v10 = src[y1 * sw + x0];
            T v11 = src[y1 * sw + x1];
            dst[i * dw + j] = (1 - fy) * ((1 - fx) * v00 + fx * v01)
                            + fy       * ((1 - fx) * v10 + fx * v11);
        }
    }
}

template<typename T>
SubpixelTOCorrectionCPU<T>::SubpixelTOCorrectionCPU(
    int H, int W, T threshold, T g_sigma, int cpu_nthreads, int n)
    : img_height(H), img_width(W), interp_n((n < 0) ? 0 : n), thresh(threshold), sigma(g_sigma), omp_threads(cpu_nthreads)
{
    neighbor_radius = static_cast<T>(2);
    output_dir = "./output_files";
    num_of_edge_data = 4;
    edge_pt_list_idx = 0;
    time_nms = time_conv = time_correct = 0;

    int scale = 1 << interp_n;
    work_height = (img_height - 1) * scale + 1;
    work_width  = (img_width  - 1) * scale + 1;

    kernel_rad = static_cast<int>(std::ceil(sigma * 4));

    edgemap = theta = dirx = diry = nullptr;
    Hx = Hy = Px = Py = Pxx = Pxy = Pyy = Hxx = Hxy = Hyy = nullptr;
    conv_tmp = nullptr;
    nms_x = nms_y = nms_mag = nullptr;
    nms_valid = nullptr;
    G_1d = dG_1d = d2G_1d = nullptr;
    subpix_edge_pts_final = nullptr;
    TO_edgemap = TO_orientation = nullptr;

    allocate_work_buffers(work_height, work_width);
    build_gaussian_kernels();

    TO_edgemap = new T[img_height * img_width];
    TO_orientation = new T[img_height * img_width];
    subpix_edge_pts_final = new T[work_height * work_width * num_of_edge_data];
}

template<typename T>
void SubpixelTOCorrectionCPU<T>::allocate_work_buffers(int h, int w)
{
    free_work_buffers();
    work_height = h;
    work_width = w;
    const int n = h * w;

    edgemap = new T[n];
    theta   = new T[n];
    dirx    = new T[n];
    diry    = new T[n];
    Hx      = new T[n];
    Hy      = new T[n];
    Px      = new T[n];
    Py      = new T[n];
    Pxx     = new T[n];
    Pxy     = new T[n];
    Pyy     = new T[n];
    Hxx     = new T[n];
    Hxy     = new T[n];
    Hyy     = new T[n];
    conv_tmp = new T[n];
    nms_x   = new T[n];
    nms_y   = new T[n];
    nms_mag = new T[n];
    nms_valid = new unsigned char[n];

    std::fill(edgemap, edgemap + n, 0);
    std::fill(theta, theta + n, 0);
}

template<typename T>
void SubpixelTOCorrectionCPU<T>::free_work_buffers()
{
    delete[] edgemap; edgemap = nullptr;
    delete[] theta;   theta = nullptr;
    delete[] dirx;    dirx = nullptr;
    delete[] diry;    diry = nullptr;
    delete[] Hx;      Hx = nullptr;
    delete[] Hy;      Hy = nullptr;
    delete[] Px;      Px = nullptr;
    delete[] Py;      Py = nullptr;
    delete[] Pxx;     Pxx = nullptr;
    delete[] Pxy;     Pxy = nullptr;
    delete[] Pyy;     Pyy = nullptr;
    delete[] Hxx;     Hxx = nullptr;
    delete[] Hxy;     Hxy = nullptr;
    delete[] Hyy;     Hyy = nullptr;
    delete[] conv_tmp; conv_tmp = nullptr;
    delete[] nms_x;   nms_x = nullptr;
    delete[] nms_y;   nms_y = nullptr;
    delete[] nms_mag; nms_mag = nullptr;
    delete[] nms_valid; nms_valid = nullptr;
}

template<typename T>
SubpixelTOCorrectionCPU<T>::~SubpixelTOCorrectionCPU()
{
    free_work_buffers();
    delete[] G_1d;
    delete[] dG_1d;
    delete[] d2G_1d;
    delete[] subpix_edge_pts_final;
    delete[] TO_edgemap;
    delete[] TO_orientation;
}

template<typename T>
void SubpixelTOCorrectionCPU<T>::build_gaussian_kernels()
{
    const int ksz = 2 * kernel_rad + 1;
    delete[] G_1d;   G_1d = new T[ksz];
    delete[] dG_1d;  dG_1d = new T[ksz];
    delete[] d2G_1d; d2G_1d = new T[ksz];

    const T s2 = sigma * sigma;
    const T s3 = s2 * sigma;
    const T s5 = s2 * s3;
    const T nrm = std::sqrt(static_cast<T>(2) * M_PI);

    for (int k = -kernel_rad; k <= kernel_rad; k++) {
        T kk = static_cast<T>(k);
        T e = std::exp(-(kk * kk) / (static_cast<T>(2) * s2));
        G_1d[k + kernel_rad]   = e / (nrm * sigma);
        dG_1d[k + kernel_rad]  = -kk * e / (nrm * s3);
        d2G_1d[k + kernel_rad] = (kk * kk - s2) * e / (nrm * s5);
    }
}

template<typename T>
T SubpixelTOCorrectionCPU<T>::wrap_mod_pi(T a) const
{
    T r = std::fmod(a, M_PI);
    if (r < 0) r += M_PI;
    return r;
}

template<typename T>
void SubpixelTOCorrectionCPU<T>::set_edgemap(const T *data)
{
    if (interp_n == 0) {
        std::copy(data, data + img_height * img_width, edgemap);
    } else {
        bilinear_resize(data, img_height, img_width, edgemap, work_height, work_width);
    }
}

template<typename T>
void SubpixelTOCorrectionCPU<T>::set_orientation(const T *data)
{
    if (interp_n == 0) {
        std::copy(data, data + img_height * img_width, theta);
    } else {
        bilinear_resize(data, img_height, img_width, theta, work_height, work_width);
    }
}

#if OPENCV_SUPPORT
template<typename T>
static void mat_to_array(const cv::Mat &image, T *dst, int h, int w, bool as_probability)
{
    cv::Mat gray;
    if (image.channels() > 1)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image;

    cv::Mat resized = gray;
    if (gray.rows != h || gray.cols != w)
        cv::resize(gray, resized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            T v;
            switch (resized.depth()) {
            case CV_8U:  v = static_cast<T>(resized.at<unsigned char>(i, j));
                         if (as_probability) v /= static_cast<T>(255); break;
            case CV_16U: v = static_cast<T>(resized.at<unsigned short>(i, j));
                         if (as_probability) v /= static_cast<T>(65535); break;
            case CV_32F: v = static_cast<T>(resized.at<float>(i, j)); break;
            case CV_64F: v = static_cast<T>(resized.at<double>(i, j)); break;
            default:     v = static_cast<T>(resized.at<unsigned char>(i, j)); break;
            }
            dst[i * w + j] = v;
        }
    }
}

template<typename T>
void SubpixelTOCorrectionCPU<T>::set_edgemap(const cv::Mat &image)
{
    std::vector<T> buf(img_height * img_width);
    mat_to_array(image, buf.data(), img_height, img_width, true);
    set_edgemap(buf.data());
}

template<typename T>
void SubpixelTOCorrectionCPU<T>::set_orientation(const cv::Mat &image)
{
    std::vector<T> buf(img_height * img_width);
    mat_to_array(image, buf.data(), img_height, img_width, false);
    if (image.depth() == CV_8U) {
        for (int k = 0; k < img_height * img_width; k++)
            buf[k] = buf[k] * M_PI / static_cast<T>(255);
    }
    set_orientation(buf.data());
}
#endif

template<typename T>
void SubpixelTOCorrectionCPU<T>::compute_dir_from_theta()
{
    const int n = work_height * work_width;
    for (int k = 0; k < n; k++) {
        T a = wrap_mod_pi(theta[k]);
        dirx[k] = std::cos(a);
        diry[k] = std::sin(a);
    }
}

// Dollar / edgesDetect orientation from a ridge map:
//   [Ox,Oy]=gradient2(convTri(E,4));
//   [Oxx,~]=gradient2(Ox); [Oxy,Oyy]=gradient2(Oy);
//   O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),M_PI);
template<typename T>
void SubpixelTOCorrectionCPU<T>::estimate_orientation()
{
    const int h = work_height;
    const int w = work_width;
    const int r = 4;
    const int ksz = 2 * r + 1;
    std::vector<T> tri(ksz);
    T tri_sum = 0;
    for (int k = -r; k <= r; k++) {
        tri[k + r] = static_cast<T>(r + 1 - std::abs(k));
        tri_sum += tri[k + r];
    }
    for (int k = 0; k < ksz; k++)
        tri[k] /= tri_sum;

    std::vector<T> tmp(h * w), sm(h * w);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            T acc = 0;
            for (int q = -r; q <= r; q++)
                acc += Emap(i, wrap_symmetric(j - q, w)) * tri[q + r];
            tmp[i * w + j] = acc;
        }
    }
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            T acc = 0;
            for (int p = -r; p <= r; p++)
                acc += tmp[wrap_symmetric(i - p, h) * w + j] * tri[p + r];
            sm[i * w + j] = acc;
        }
    }

    auto deriv_x = [&](const T *a, int i, int j) -> T {
        if (j == 0) return a[i * w + 1] - a[i * w + 0];
        if (j == w - 1) return a[i * w + (w - 1)] - a[i * w + (w - 2)];
        return static_cast<T>(0.5) * (a[i * w + (j + 1)] - a[i * w + (j - 1)]);
    };
    auto deriv_y = [&](const T *a, int i, int j) -> T {
        if (i == 0) return a[1 * w + j] - a[0 * w + j];
        if (i == h - 1) return a[(h - 1) * w + j] - a[(h - 2) * w + j];
        return static_cast<T>(0.5) * (a[(i + 1) * w + j] - a[(i - 1) * w + j]);
    };

    std::vector<T> Ox(h * w), Oy(h * w);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            Ox[i * w + j] = deriv_x(sm.data(), i, j);
            Oy[i * w + j] = deriv_y(sm.data(), i, j);
        }
    }

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            T Oxx = deriv_x(Ox.data(), i, j);
            T Oxy = deriv_x(Oy.data(), i, j);
            T Oyy = deriv_y(Oy.data(), i, j);
            T sgn = (Oxy < 0) ? T(1) : ((Oxy > 0) ? T(-1) : T(0));
            T ang = std::atan(Oyy * sgn / (Oxx + static_cast<T>(1e-5)));
            theta[i * w + j] = wrap_mod_pi(ang);
        }
    }
}

template<typename T>
void SubpixelTOCorrectionCPU<T>::conv_separable_circular(const T *src, T *dst,
                                                         const T *kx, const T *ky)
{
    const int h = work_height;
    const int w = work_width;
    const int rad = kernel_rad;

    omp_set_num_threads(omp_threads);
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                T acc = 0;
                for (int q = -rad; q <= rad; q++) {
                    int jj = wrap_idx(j - q, w);
                    acc += src[i * w + jj] * kx[q + rad];
                }
                conv_tmp[i * w + j] = acc;
            }
        }
        #pragma omp for schedule(static)
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                T acc = 0;
                for (int p = -rad; p <= rad; p++) {
                    int ii = wrap_idx(i - p, h);
                    acc += conv_tmp[ii * w + j] * ky[p + rad];
                }
                dst[i * w + j] = acc;
            }
        }
    }
}

template<typename T>
void SubpixelTOCorrectionCPU<T>::nms_token(int margin)
{
    const int h = work_height;
    const int w = work_width;
    const int n = h * w;
    std::fill(nms_x, nms_x + n, 0);
    std::fill(nms_y, nms_y + n, 0);
    std::fill(nms_mag, nms_mag + n, 0);
    std::fill(nms_valid, nms_valid + n, 0);

    const int j0 = margin + 1;
    const int j1 = w - (margin + 2);
    const int i0 = margin + 1;
    const int i1 = h - (margin + 2);

    omp_set_num_threads(omp_threads);
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        T gx, gy, mag_dir, norm_dir_x, norm_dir_y;
        T slope, fp, fm, f;
        T coeff_A, coeff_B, coeff_C, s, s_star, max_f;

        #pragma omp for schedule(dynamic)
        for (int j = j0; j < j1; j++) {
            for (int i = i0; i < i1; i++) {
                if (Emap(i, j) <= thresh)
                    continue;

                gx = Dirx(i, j);
                gy = Diry(i, j);
                if (std::abs(gx) < static_cast<T>(1e-5) && std::abs(gy) < static_cast<T>(1e-5))
                    continue;

                mag_dir = std::sqrt(gx * gx + gy * gy);
                norm_dir_x = gx / mag_dir;
                norm_dir_y = gy / mag_dir;
                f = Emap(i, j);

                if ((gx >= 0) && (gy >= 0)) {
                    if (gx >= gy) {
                        slope = norm_dir_y / norm_dir_x;
                        fp = Emap(i, j + 1) * (1 - slope) + Emap(i + 1, j + 1) * slope;
                        fm = Emap(i, j - 1) * (1 - slope) + Emap(i - 1, j - 1) * slope;
                    } else {
                        slope = norm_dir_x / norm_dir_y;
                        fp = Emap(i + 1, j) * (1 - slope) + Emap(i + 1, j + 1) * slope;
                        fm = Emap(i - 1, j) * (1 - slope) + Emap(i - 1, j - 1) * slope;
                    }
                } else if ((gx < 0) && (gy >= 0)) {
                    if (std::abs(gx) < gy) {
                        slope = -norm_dir_x / norm_dir_y;
                        fp = Emap(i + 1, j) * (1 - slope) + Emap(i + 1, j - 1) * slope;
                        fm = Emap(i - 1, j) * (1 - slope) + Emap(i - 1, j + 1) * slope;
                    } else {
                        slope = -norm_dir_y / norm_dir_x;
                        fp = Emap(i, j - 1) * (1 - slope) + Emap(i + 1, j - 1) * slope;
                        fm = Emap(i, j + 1) * (1 - slope) + Emap(i - 1, j + 1) * slope;
                    }
                } else if ((gx < 0) && (gy < 0)) {
                    if (std::abs(gx) >= std::abs(gy)) {
                        slope = norm_dir_y / norm_dir_x;
                        fp = Emap(i, j - 1) * (1 - slope) + Emap(i - 1, j - 1) * slope;
                        fm = Emap(i, j + 1) * (1 - slope) + Emap(i + 1, j + 1) * slope;
                    } else {
                        slope = norm_dir_x / norm_dir_y;
                        fp = Emap(i - 1, j) * (1 - slope) + Emap(i - 1, j - 1) * slope;
                        fm = Emap(i + 1, j) * (1 - slope) + Emap(i + 1, j + 1) * slope;
                    }
                } else {
                    if (gx < std::abs(gy)) {
                        slope = -norm_dir_x / norm_dir_y;
                        fp = Emap(i - 1, j) * (1 - slope) + Emap(i - 1, j + 1) * slope;
                        fm = Emap(i + 1, j) * (1 - slope) + Emap(i + 1, j - 1) * slope;
                    } else {
                        slope = -norm_dir_y / norm_dir_x;
                        fp = Emap(i, j + 1) * (1 - slope) + Emap(i - 1, j + 1) * slope;
                        fm = Emap(i, j - 1) * (1 - slope) + Emap(i + 1, j - 1) * slope;
                    }
                }

                if (!((f > fm && f > fp) || (f > fm && f >= fp) || (f >= fm && f > fp)))
                    continue;

                s = std::sqrt(1 + slope * slope);
                coeff_A = (fm + fp - 2 * f) / (2 * s * s);
                coeff_B = (fp - fm) / (2 * s);
                coeff_C = f;
                if (std::abs(coeff_A) < static_cast<T>(1e-16))
                    continue;

                s_star = -coeff_B / (2 * coeff_A);
                max_f = coeff_A * s_star * s_star + coeff_B * s_star + coeff_C;

                if (std::abs(s_star) <= std::sqrt(static_cast<T>(2))) {
                    nms_x_(i, j) = static_cast<T>(j) + s_star * norm_dir_x;
                    nms_y_(i, j) = static_cast<T>(i) + s_star * norm_dir_y;
                    nms_mag_(i, j) = std::abs(max_f);
                    nms_valid_(i, j) = 1;
                }
            }
        }
    }
    time_nms = omp_get_wtime() - start;
    std::cout << "- Time of NMS (OpenMP): " << time_nms * 1000 << " (ms)" << std::endl;
}

template<typename T>
T SubpixelTOCorrectionCPU<T>::interp2_cubic(const T *src, T x, T y) const
{
    const int h = work_height;
    const int w = work_width;
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    T fx = x - x0;
    T fy = y - y0;

    T wx[4], wy[4];
    wx[0] = cubic_kernel(fx + 1);
    wx[1] = cubic_kernel(fx);
    wx[2] = cubic_kernel(fx - 1);
    wx[3] = cubic_kernel(fx - 2);
    wy[0] = cubic_kernel(fy + 1);
    wy[1] = cubic_kernel(fy);
    wy[2] = cubic_kernel(fy - 1);
    wy[3] = cubic_kernel(fy - 2);

    auto sample = [&](int ii, int jj) -> T {
        ii = (ii < 0) ? 0 : (ii >= h ? h - 1 : ii);
        jj = (jj < 0) ? 0 : (jj >= w ? w - 1 : jj);
        return src[ii * w + jj];
    };

    T acc = 0;
    for (int p = 0; p < 4; p++) {
        T row = 0;
        int ii = y0 - 1 + p;
        for (int q = 0; q < 4; q++)
            row += sample(ii, x0 - 1 + q) * wx[q];
        acc += row * wy[p];
    }
    return acc;
}

template<typename T>
struct TOEdgel {
    T x, y, ori, mag;
};

template<typename T>
static int remove_isolated_edgels(std::vector<TOEdgel<T> >& edges, T radius)
{
    const int n0 = static_cast<int>(edges.size());
    if (radius <= 0 || n0 < 2)
        return 0;

    const T r2 = radius * radius;
    const T inv = 1 / radius;
    std::unordered_map<long long, std::vector<int> > grid;
    grid.reserve(static_cast<size_t>(n0) * 2);

    auto cell_key = [](int cx, int cy) -> long long {
        return (static_cast<long long>(static_cast<unsigned int>(cx)) << 32)
             | static_cast<unsigned int>(cy);
    };

    std::vector<int> cx(n0), cy(n0);
    for (int i = 0; i < n0; i++) {
        cx[i] = static_cast<int>(std::floor(edges[i].x * inv));
        cy[i] = static_cast<int>(std::floor(edges[i].y * inv));
        grid[cell_key(cx[i], cy[i])].push_back(i);
    }

    std::vector<char> keep(n0, 0);
    for (int i = 0; i < n0; i++) {
        bool found = false;
        for (int dy = -1; dy <= 1 && !found; dy++) {
            for (int dx = -1; dx <= 1 && !found; dx++) {
                auto it = grid.find(cell_key(cx[i] + dx, cy[i] + dy));
                if (it == grid.end())
                    continue;
                for (size_t k = 0; k < it->second.size(); k++) {
                    int j = it->second[k];
                    if (j == i)
                        continue;
                    T ddx = edges[i].x - edges[j].x;
                    T ddy = edges[i].y - edges[j].y;
                    if (ddx * ddx + ddy * ddy <= r2) {
                        found = true;
                        break;
                    }
                }
            }
        }
        keep[i] = found ? 1 : 0;
    }

    int w = 0;
    int removed = 0;
    for (int i = 0; i < n0; i++) {
        if (keep[i])
            edges[w++] = edges[i];
        else
            removed++;
    }
    edges.resize(w);
    return removed;
}

template<typename T>
int SubpixelTOCorrectionCPU<T>::run(T *edginfo_out)
{
    compute_dir_from_theta();

    int margin = 3;
    if (interp_n > 0)
        margin *= (1 << interp_n);

    nms_token(margin);

    const int n = work_height * work_width;
    for (int k = 0; k < n; k++) {
        Hx[k] = edgemap[k] * dirx[k];
        Hy[k] = edgemap[k] * diry[k];
    }

    double start = omp_get_wtime();
    conv_separable_circular(edgemap, Px,  dG_1d,  G_1d);
    conv_separable_circular(edgemap, Py,  G_1d,   dG_1d);
    conv_separable_circular(edgemap, Pxx, d2G_1d, G_1d);
    conv_separable_circular(edgemap, Pxy, dG_1d,  dG_1d);
    conv_separable_circular(edgemap, Pyy, G_1d,   d2G_1d);
    conv_separable_circular(Hx,      Hxx, dG_1d,  G_1d);
    conv_separable_circular(Hy,      Hyy, G_1d,   dG_1d);
    conv_separable_circular(Hy,      Hxy, dG_1d,  G_1d);
    time_conv = omp_get_wtime() - start;
    std::cout << "- Time of Gaussian derivatives (OpenMP): " << time_conv * 1000
              << " (ms)" << std::endl;

    std::fill(TO_edgemap, TO_edgemap + img_height * img_width, 0);
    std::fill(TO_orientation, TO_orientation + img_height * img_width, 0);
    std::fill(subpix_edge_pts_final, subpix_edge_pts_final + n * num_of_edge_data, 0);

    const T coord_scale = static_cast<T>(1 << interp_n);
    start = omp_get_wtime();

    std::vector<TOEdgel<T> > local_edges;
    local_edges.reserve(n / 16);

    for (int i = 0; i < work_height; i++) {
        for (int j = 0; j < work_width; j++) {
            if (!nms_valid_(i, j))
                continue;

            T sx = nms_x_(i, j);
            T sy = nms_y_(i, j);
            T mag_e = nms_mag_(i, j);

            T Px_e  = interp2_cubic(Px,  sx, sy);
            T Py_e  = interp2_cubic(Py,  sx, sy);
            T Pxx_e = interp2_cubic(Pxx, sx, sy);
            T Pyy_e = interp2_cubic(Pyy, sx, sy);
            T Pxy_e = interp2_cubic(Pxy, sx, sy);
            T Hx_e  = interp2_cubic(Hx,  sx, sy);
            T Hxx_e = interp2_cubic(Hxx, sx, sy);
            T Hxy_e = interp2_cubic(Hxy, sx, sy);
            T Hyy_e = interp2_cubic(Hyy, sx, sy);
            T Hy_e  = interp2_cubic(Hy,  sx, sy);

            T Fx_e = Px_e * Hxx_e + Pxx_e * Hx_e + Py_e * Hxy_e + Pxy_e * Hy_e;
            T Fy_e = Px_e * Hxy_e + Pxy_e * Hx_e + Py_e * Hyy_e + Pyy_e * Hy_e;
            T F_mag = std::sqrt(Fx_e * Fx_e + Fy_e * Fy_e);

            T ori;
            if (F_mag < static_cast<T>(1e-12)) {
                ori = std::atan2(Dirx(i, j), -Diry(i, j));
            } else {
                Fx_e /= F_mag;
                Fy_e /= F_mag;
                ori = std::atan2(Fx_e, -Fy_e);
            }

            T x0 = sx / coord_scale;
            T y0 = sy / coord_scale;
            local_edges.push_back({x0, y0, ori, mag_e});
        }
    }

    int n_isolated = remove_isolated_edgels(local_edges, neighbor_radius);
    if (neighbor_radius > 0) {
        std::cout << "- Removed " << n_isolated << " isolated edgels (no neighbor within "
                  << neighbor_radius << " px)" << std::endl;
    }

    for (size_t k = 0; k < local_edges.size(); k++) {
        int X = static_cast<int>(std::round(local_edges[k].x));
        int Y = static_cast<int>(std::round(local_edges[k].y));
        X = std::max(0, std::min(X, img_width - 1));
        Y = std::max(0, std::min(Y, img_height - 1));
        TO_E(Y, X) = local_edges[k].mag;
        TO_O(Y, X) = local_edges[k].ori;
    }

    edge_pt_list_idx = static_cast<int>(local_edges.size());
    for (int k = 0; k < edge_pt_list_idx; k++) {
        edg(k, 0) = local_edges[k].x;
        edg(k, 1) = local_edges[k].y;
        edg(k, 2) = local_edges[k].ori;
        edg(k, 3) = local_edges[k].mag;
        if (edginfo_out) {
            out_e(k, 0) = local_edges[k].x;
            out_e(k, 1) = local_edges[k].y;
            out_e(k, 2) = local_edges[k].ori;
            out_e(k, 3) = local_edges[k].mag;
        }
    }

    time_correct = omp_get_wtime() - start;
    std::cout << "- Time of TO orientation correction: " << time_correct * 1000
              << " (ms)" << std::endl;

    return edge_pt_list_idx;
}

template<typename T>
void SubpixelTOCorrectionCPU<T>::set_output_dir(const std::string &dir)
{
    output_dir = dir.empty() ? "./output_files" : dir;
}

template<typename T>
void SubpixelTOCorrectionCPU<T>::set_neighbor_radius(T radius)
{
    neighbor_radius = radius;
}

template<typename T>
void SubpixelTOCorrectionCPU<T>::write_array_to_file(std::string filename, T *wr_data,
                                                     int first_dim, int second_dim)
{
#define wr_data(i, j) wr_data[(i) * second_dim + (j)]
    std::cout << "writing data to a file " << filename << " ..." << std::endl;
    if (wr_data == nullptr || first_dim <= 0 || second_dim <= 0) {
        std::cout << "write data file skipped: invalid buffer or dimensions." << std::endl;
        return;
    }
    std::string out_file_name = output_dir;
    if (!out_file_name.empty() && out_file_name.back() != '/')
        out_file_name.push_back('/');
    out_file_name.append(filename);
    std::ofstream out_file(out_file_name);
    if (!out_file.is_open()) {
        std::cout << "write data file cannot be opened!" << std::endl;
        return;
    }
    for (int i = 0; i < first_dim; i++) {
        for (int j = 0; j < second_dim; j++)
            out_file << wr_data(i, j) << "\t";
        out_file << "\n";
    }
    out_file.close();
#undef wr_data
}

template class SubpixelTOCorrectionCPU<double>;
template class SubpixelTOCorrectionCPU<float>;

#undef Emap
#undef Theta
#undef Dirx
#undef Diry
#undef Hx_
#undef Hy_
#undef Px_
#undef Py_
#undef Pxx_
#undef Pxy_
#undef Pyy_
#undef Hxx_
#undef Hxy_
#undef Hyy_
#undef nms_x_
#undef nms_y_
#undef nms_mag_
#undef nms_valid_
#undef TO_E
#undef TO_O
#undef edg
#undef out_e

#endif
