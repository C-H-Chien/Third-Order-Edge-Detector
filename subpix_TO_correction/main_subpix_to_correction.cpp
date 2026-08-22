// ======================================================================================
// Subpixel third-order correction on a dense edge map (CPU / OpenMP)
//
// Usage:
//   ./TO_correct <edgemap> [orientation] [thresh] [sigma] [nthreads] [output_dir] [nbr_radius]
//
//   edgemap      dense edge strength / probability map (image or whitespace .txt)
//   orientation  optional orientation map in radians (image or .txt). If omitted,
//                it is estimated from the edge map (Dollar Hessian method).
//   thresh       NMS threshold on the edge map (default 0.2)
//   sigma        Gaussian scale of the TO operator (default 2)
//   nthreads     OpenMP threads (default 1)
//   output_dir   output directory (default ./output_files)
//   nbr_radius   drop edgels with no neighbor within this radius (default 2; 0 disables)
//
// Output (0-based pixel coordinates, same layout as TOED_edges.txt):
//   TO_corrected_edges.txt   N x 4 : x, y, orientation, strength
// ======================================================================================
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <vector>
#include <cctype>

#include "indices.hpp"
#include "subpix_to_correction.hpp"

#if OPENCV_SUPPORT
#include <opencv2/opencv.hpp>
#endif

static bool mkdir_p(const std::string& path)
{
    if (path.empty() || path == "." || path == "/")
        return true;

    struct stat st;
    if (stat(path.c_str(), &st) == 0)
        return S_ISDIR(st.st_mode);

    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos && pos > 0) {
        if (!mkdir_p(path.substr(0, pos)))
            return false;
    }

    return (mkdir(path.c_str(), 0755) == 0) || (errno == EEXIST);
}

static bool file_exists(const std::string& path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static bool is_number(const char *s)
{
    if (s == nullptr || *s == '\0')
        return false;
    char *end = nullptr;
    std::strtod(s, &end);
    return end != s && *end == '\0';
}

static bool load_txt_matrix(const std::string& path, std::vector<double>& data,
                            int& height, int& width)
{
    std::ifstream in(path);
    if (!in.is_open())
        return false;

    std::vector<std::vector<double> > rows;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        double v;
        while (iss >> v)
            row.push_back(v);
        if (!row.empty())
            rows.push_back(row);
    }
    if (rows.empty())
        return false;

    height = static_cast<int>(rows.size());
    width  = static_cast<int>(rows[0].size());
    data.resize(height * width);
    for (int i = 0; i < height; i++) {
        if (static_cast<int>(rows[i].size()) != width)
            return false;
        for (int j = 0; j < width; j++)
            data[i * width + j] = rows[i][j];
    }
    return true;
}

static bool is_txt_path(const std::string& path)
{
    if (path.size() < 4)
        return false;
    std::string ext = path.substr(path.size() - 4);
    for (size_t i = 0; i < ext.size(); i++)
        ext[i] = static_cast<char>(std::tolower(ext[i]));
    return ext == ".txt" || ext == ".dat" || ext == ".edg";
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0]
                  << " <edgemap> [orientation] [thresh] [sigma] [nthreads] [output_dir] [nbr_radius]\n"
                  << "  edgemap      dense edge strength/probability map (image or .txt)\n"
                  << "  orientation  optional orientation map in radians (image or .txt)\n"
                  << "               If omitted, estimated from the edge map.\n"
                  << "  thresh       NMS threshold (default 0.2)\n"
                  << "  sigma        Gaussian sigma (default 2)\n"
                  << "  nthreads     OpenMP threads (default 1)\n"
                  << "  output_dir   output directory (default ./output_files)\n"
                  << "  nbr_radius   drop edgels with no neighbor within this radius (default 2; 0 disables)\n";
        return 0;
    }

    std::string edgemap_path(argv[1]);
    if (!file_exists(edgemap_path)) {
        std::cout << "File " << edgemap_path << " not found." << std::endl;
        return 0;
    }

    int argi = 2;
    std::string orient_path;
    if (argc > 2 && !is_number(argv[2])) {
        orient_path = argv[2];
        if (!file_exists(orient_path)) {
            std::cerr << "Error: orientation file not found: " << orient_path << std::endl;
            return 1;
        }
        argi = 3;
    }

    double thresh = 0.2;
    double sigma = 2.0;
    int nthreads = 1;
    std::string output_dir = "./output_files";
    double nbr_radius = 2.0;

    if (argc > argi) { thresh = std::atof(argv[argi]); argi++; }
    if (argc > argi) { sigma = std::atof(argv[argi]); argi++; }
    if (argc > argi) { nthreads = std::atoi(argv[argi]); argi++; }
    if (argc > argi) { output_dir = argv[argi]; argi++; }
    if (argc > argi) { nbr_radius = std::atof(argv[argi]); argi++; }

    if (nthreads < 1) nthreads = 1;
    if (sigma <= 0) {
        std::cerr << "Error: sigma must be positive." << std::endl;
        return 1;
    }
    if (!mkdir_p(output_dir)) {
        std::cerr << "Error: failed to create output directory: " << output_dir << std::endl;
        return 1;
    }

    int height = 0, width = 0;
    std::vector<double> E;
    std::vector<double> O;
    bool have_orient = !orient_path.empty();

    if (is_txt_path(edgemap_path)) {
        if (!load_txt_matrix(edgemap_path, E, height, width)) {
            std::cerr << "Error: failed to read edgemap text file " << edgemap_path << std::endl;
            return 1;
        }
    } else {
#if OPENCV_SUPPORT
        cv::Mat img = cv::imread(edgemap_path, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            std::cerr << "Error: Failed to load image " << edgemap_path << std::endl;
            return 1;
        }
        height = img.rows;
        width  = img.cols;
        E.resize(height * width);
        cv::Mat gray;
        if (img.channels() > 1)
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        else
            gray = img;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                double v;
                switch (gray.depth()) {
                case CV_8U:  v = gray.at<unsigned char>(i, j) / 255.0; break;
                case CV_16U: v = gray.at<unsigned short>(i, j) / 65535.0; break;
                case CV_32F: v = gray.at<float>(i, j); break;
                case CV_64F: v = gray.at<double>(i, j); break;
                default:     v = gray.at<unsigned char>(i, j) / 255.0; break;
                }
                E[i * width + j] = v;
            }
        }
#else
        std::cerr << "Error: OpenCV is disabled; edgemap must be a .txt file." << std::endl;
        return 1;
#endif
    }

    if (have_orient) {
        int oh = 0, ow = 0;
        if (is_txt_path(orient_path)) {
            if (!load_txt_matrix(orient_path, O, oh, ow)) {
                std::cerr << "Error: failed to read orientation text file " << orient_path << std::endl;
                return 1;
            }
        } else {
#if OPENCV_SUPPORT
            cv::Mat oimg = cv::imread(orient_path, cv::IMREAD_UNCHANGED);
            if (oimg.empty()) {
                std::cerr << "Error: Failed to load orientation image " << orient_path << std::endl;
                return 1;
            }
            oh = oimg.rows;
            ow = oimg.cols;
            O.resize(oh * ow);
            cv::Mat gray;
            if (oimg.channels() > 1)
                cv::cvtColor(oimg, gray, cv::COLOR_BGR2GRAY);
            else
                gray = oimg;
            const double PI = 3.14159265358979323846;
            for (int i = 0; i < oh; i++) {
                for (int j = 0; j < ow; j++) {
                    double v;
                    switch (gray.depth()) {
                    case CV_8U:  v = gray.at<unsigned char>(i, j) * PI / 255.0; break;
                    case CV_16U: v = gray.at<unsigned short>(i, j) * PI / 65535.0; break;
                    case CV_32F: v = gray.at<float>(i, j); break;
                    case CV_64F: v = gray.at<double>(i, j); break;
                    default:     v = gray.at<unsigned char>(i, j) * PI / 255.0; break;
                    }
                    O[i * ow + j] = v;
                }
            }
#else
            std::cerr << "Error: OpenCV is disabled; orientation must be a .txt file." << std::endl;
            return 1;
#endif
        }
        if (oh != height || ow != width) {
            std::cerr << "Error: orientation size (" << oh << "x" << ow
                      << ") does not match edgemap (" << height << "x" << width << ")." << std::endl;
            return 1;
        }
    }

    std::cout << "Output directory: " << output_dir << std::endl;
    std::cout << "Edgemap: " << height << " x " << width
              << ", thresh=" << thresh << ", sigma=" << sigma
              << ", threads=" << nthreads
              << ", nbr_radius=" << nbr_radius << std::endl;
    if (have_orient)
        std::cout << "Orientation: loaded from " << orient_path << std::endl;
    else
        std::cout << "Orientation: estimated from edgemap" << std::endl;

    double *edginfo = new double[height * width * 4];
    for (int i = 0; i < height * width * 4; i++)
        edginfo[i] = 0;

    printf("############################################\n");
    printf("##   Subpixel TO Correction (CPU/OpenMP) ##\n");
    printf("############################################\n");

    SubpixelTOCorrectionCPU<double> to_corr(height, width, thresh, sigma, nthreads, 0);
    to_corr.set_output_dir(output_dir);
    to_corr.set_neighbor_radius(nbr_radius);
    to_corr.set_edgemap(E.data());
    if (have_orient)
        to_corr.set_orientation(O.data());
    else
        to_corr.estimate_orientation();

    int edge_num = to_corr.run(edginfo);
    std::cout << "Number of corrected edgels = " << edge_num << std::endl;

    to_corr.write_array_to_file("TO_corrected_edges.txt", edginfo, edge_num, 4);
    to_corr.write_array_to_file("TO_edgemap.txt", to_corr.TO_edgemap, height, width);
    to_corr.write_array_to_file("TO_orientation.txt", to_corr.TO_orientation, height, width);

#if OPENCV_SUPPORT
    cv::Mat vis(height, width, CV_8UC1);
    double vmax = 0;
    for (int k = 0; k < height * width; k++)
        vmax = std::max(vmax, to_corr.TO_edgemap[k]);
    if (vmax < 1e-12) vmax = 1;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double v = to_corr.TO_edgemap[i * width + j] / vmax;
            vis.at<unsigned char>(i, j) = static_cast<unsigned char>(std::max(0.0, std::min(255.0, v * 255.0)));
        }
    }
    std::string vis_path = output_dir;
    if (!vis_path.empty() && vis_path.back() != '/')
        vis_path.push_back('/');
    vis_path += "TO_edgemap.png";
    cv::imwrite(vis_path, vis);
    std::cout << "wrote " << vis_path << std::endl;
#endif

    delete[] edginfo;
    return 0;
}
