#ifndef TOED_CONFIG_HPP
#define TOED_CONFIG_HPP
// =============================================================================
// Central configuration for Third-Order Edge Detection (CPU / GPU / curvelets).
// Edit this file to control feature flags and algorithm parameters.
// =============================================================================

#include <cmath>

// ----------------------------------------------------------------------------
// Feature flags (compile-time)
// ----------------------------------------------------------------------------
#define OPENCV_SUPPORT                      (true)

// Enable curvelet / curvel formation (double precision only)
#define CurvelFormation                     (false)

// Precision selection (used by main_gpu_cpu.cpp)
#define Use_Double_Precision                (true)
#define Use_Single_Precision                (false)

// Write intermediate arrays (Ix, Iy, grad mag, etc.) for debugging
#define WriteDataToFile                     (0)

// Write third-order edges as an `.edg` file (EDGE_MAP v3.0, MATLAB-compatible)
#define WriteEdgFile                        (true)

// ----------------------------------------------------------------------------
// Algorithm / runtime parameters
// ----------------------------------------------------------------------------
namespace toed_cfg {

// ====================== Third-order edge detector =============================
// Gaussian scale. CPU filters are built from this value at runtime
// GPU convolution still uses hardcoded kernels generated for sigma = 2
constexpr int    sigma           = 1;

// Filter support. CPU kernels have length kernel_size (unshifted) and
// kernel_size+2 (half-pixel shift). Use 2*ceil(4*sigma)+1 (17 when sigma = 2)
constexpr int    kernel_size     = 17;

// Gradient-magnitude threshold in NMS (keep edges with mag > thresh)
// Matches MATLAB: grad_mag > threshold.
constexpr double grad_mag_thresh = 1.0;

// Border (in interpolated pixels) skipped during NMS.
// MATLAB uses (margin+2) with margin = ceil(4*sigma); for sigma = 2 that is 10
constexpr int    nms_border      = 10;

// ====================== CLI defaults (overridable by argv) ======================
constexpr int         default_nthreads   = 1;
constexpr int         default_gpu_id     = 0;
constexpr const char* default_output_dir = "./output_files";

// ====================== Output filenames =====================================
constexpr const char* edges_txt_filename          = "TOED_edges.txt";
constexpr const char* edges_edg_filename          = "TOED_edges.edg";
constexpr const char* edges_edg_gpu_dp_filename   = "TOED_edges_gpu_dp64.edg";
constexpr const char* edges_edg_gpu_sp_filename   = "TOED_edges_gpu_sp32.edg";
constexpr const char* curvelet_chain_filename     = "chain.txt";
constexpr const char* curvelet_info_filename      = "info.txt";

// ====================== Curvelet / curvel formation (when CurvelFormation is true) ======================
constexpr int      edge_data_sz                = 4;
constexpr double   curvelet_nrad               = 3.5;
constexpr double   curvelet_gap                = 1.5;
constexpr double   curvelet_dx                 = 0.4;
constexpr double   curvelet_dt_deg             = 15.0;  // converted to radians below
constexpr double   curvelet_token_len          = 1.0;
constexpr double   curvelet_max_k              = 0.3;
constexpr unsigned curvelet_style              = 2;     // 2 = anchor-leading bidirectional
constexpr unsigned curvelet_max_size_to_group  = 4;
// 0 = curvelet map, 1 = curve fragment graph, 2 = poly arc map
constexpr unsigned curvelet_output_type        = 0;

inline double curvelet_dt_rad()
{
    return (curvelet_dt_deg / 180.0) * M_PI;
}

} // namespace toed_cfg

#endif // TOED_CONFIG_HPP
