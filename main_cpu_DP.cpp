// ======================================================================================
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
//> Change Logs
//>     Jun. 2022: Complete CPU and GPU implementation. First test on a small image.
//      Dec. 2023: Fix some bugs on CPU GPU result inconsistency issue
//      Feb. 2025: Initiate a main function for third-order edge detection (without curvelet),
//                 and it is done on CPU-only using double-precision
// ======================================================================================
#include <cmath>
#include <fstream>
#include <iterator>
#include <iostream>
#include <string>
#include <string.h>
#include <vector>
#include <stdint.h>
#include <sys/stat.h>
#include <errno.h>

#include "indices.hpp"

#if CurvelFormation
#include "curvelet/Array.hpp"
#include "curvelet/curvelet_utils.hpp"
#include "curvelet/form_curvelet_process.hpp"
#endif

#if OPENCV_SUPPORT
#include <opencv2/opencv.hpp>
#endif

// cpu
#include "cpu_toed.hpp"

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

template<typename T>
void initialize_TOED_edges( T* &TOED_edges, int height, int width ) 
{
    TOED_edges = new T[(2*height)*(2*width)*4];
    // initialization
    for (int i = 0; i < (2*height)*(2*width); i++) {
        for (int j = 0; j < 4; j++) {
            TOED_edges(i, j)  = 0;
        }
    }
}

template<typename T>
void _write_array_to_file(std::string filename, T *wr_data, int first_dim, int second_dim,
                          const std::string& output_dir)
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
        for (int j = 0; j < second_dim; j++) {
            out_file << wr_data(i, j) << "\t";
        }
        out_file << "\n";
    }

    out_file.close();
#undef wr_data
}

// Array / MATLAB column-major layout used by curvelet outputs
template<typename T>
void _write_array_to_file_colmajor(std::string filename, T *wr_data, int first_dim, int second_dim,
                                   const std::string& output_dir)
{
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
        for (int j = 0; j < second_dim; j++) {
            out_file << wr_data[i + first_dim * j] << "\t";
        }
        out_file << "\n";
    }

    out_file.close();
}

//------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0]
                  << " <input_image> [nthreads] [output_dir]" << std::endl;
        return 0;
    }

	//> Exit if the input image file doesn't open
	std::string filename(argv[1]);
	std::ifstream infile(filename, std::ios::binary);
	if (!infile.is_open())
	{
		std::cout << "File " << filename << " not found in directory." << std::endl;
		return 0;
	}

    // Load images
    int height, width;
#if OPENCV_SUPPORT
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if ( img.empty() ) {
        std::cerr << "Error: Failed to load image " << filename << std::endl;
        return 0;
    }
    height = img.rows;
    width  = img.cols;
#else
    char type[10];
	int intensity;
	// -- Storing header information and copying into the new ouput images --
	infile >> type >> width >> height >> intensity;
#endif

	//> Read number of threads if passed through command line. It is 1 by default.
	int nthreads = 1;
	if(argc > 2) {
	    nthreads = atoi( argv[2] );
	}

    std::string output_dir = "./output_files";
    if (argc > 3) {
        output_dir = argv[3];
    }
    if (!mkdir_p(output_dir)) {
        std::cerr << "Error: failed to create output directory: " << output_dir << std::endl;
        return 1;
    }
    std::cout << "Output directory: " << output_dir << std::endl;

	//> define parameters (This could be changed to argv input arguments but now let's make it fixed)
	int kernel_size = 17;
	int sigma = 2;

    // ==================================== THIRD-ORDER EDGE DETECTOR STARTS HERE ===============================================
    int edge_num;
    double *TOED_edges;
    initialize_TOED_edges<double>( TOED_edges, height, width );

    printf("############################################\n");
    printf("##         Double Precision Test          ##\n");
    printf("############################################\n");
    printf("\n ==> CPU Test (OpenMP %d threads)  \n", nthreads);
    printf("============================================\n");
    
    ThirdOrderEdgeDetectionCPU<double> toedCPU_fp64(height, width, sigma, kernel_size, nthreads);
    toedCPU_fp64.set_output_dir(output_dir);
#if OPENCV_SUPPORT
    toedCPU_fp64.preprocessing(img);
#else
    toedCPU_fp64.preprocessing(infile);
#endif
    toedCPU_fp64.convolve_img();
    edge_num = toedCPU_fp64.non_maximum_suppresion(TOED_edges);

    std::cout << "Number of edges = " << edge_num << std::endl;

#if CurvelFormation

    // -- settings (match curvelet_construction/main.cpp) --
    const int edge_data_sz = 4;
    double nrad = 3.5;
    double gap = 1.5;
    double dx = 0.4;
    double dt = (15.0 / 180.0) * M_PI;
    double token_len = 1;
    double max_k = 0.3;
    unsigned curvelet_style = 2;   // anchor-leading bidirectional
    unsigned max_size_to_group = 4;
    //> when output_type is 0, output the curvelet map
    //  when output_type is 1, output the curve fragment graph
    //  when output_type is 2, output the poly arc map
    unsigned output_type = 0;

    // form_curvelet_process expects column-major edgeinfo (same as curvelet_construction/main.cpp)
    double *TOED_edges_cm = new double[edge_num * edge_data_sz];
    for (int i = 0; i < edge_num; i++) {
        for (int j = 0; j < edge_data_sz; j++) {
            TOED_edges_cm[j * edge_num + i] = TOED_edges[i * edge_data_sz + j];
        }
    }

    arrayd edgeinfo;
    edgeinfo._data = TOED_edges_cm;
    edgeinfo.set_h(edge_num);
    edgeinfo.set_w(edge_data_sz);

    unsigned cvlet_type = curvelet_style;
    bool bCentered_grouping = cvlet_type == 0 || cvlet_type == 1;
    bool bBidirectional_grouping = cvlet_type == 0 || cvlet_type == 2;

    form_curvelet_process curvelet_pro(edgeinfo, unsigned(height), unsigned(width),
                                       nrad, gap, dx, dt, token_len, max_k,
                                       max_size_to_group,
                                       bCentered_grouping, bBidirectional_grouping);
    curvelet_pro.execute();

    unsigned out_h, out_w, info_w;
    curvelet_pro.get_output_size(out_h, out_w, output_type);

    int *out_chain = new int[out_h * out_w];
    if (output_type == 0)
        info_w = 10;
    else if (output_type == 1)
        info_w = 1;
    else
        info_w = 12;
    double *out_info = new double[out_h * info_w];

    arrayi chain;
    chain._data = out_chain;
    chain.set_h(out_h);
    chain.set_w(out_w);
    arrayd info;
    info._data = out_info;
    info.set_h(out_h);
    info.set_w(info_w);

    curvelet_pro.get_output_arrary(chain, info, output_type);

    std::cout << "(out_h, out_w) = (" << out_h << ", " << out_w << ")" << std::endl;
    std::cout << "chain width and height: " << chain.w() << ", " << chain.h() << std::endl;
    std::cout << "info width and height: " << info.w() << ", " << info.h() << std::endl;

    delete[] TOED_edges_cm;

#endif

    //> save the third-order edges to a file
    toedCPU_fp64.write_array_to_file("TOED_edges.txt", TOED_edges, edge_num, 4);
#if CurvelFormation
    _write_array_to_file_colmajor("chain.txt", chain._data, chain.h(), chain.w(), output_dir);
    _write_array_to_file_colmajor("info.txt", info._data, info.h(), info.w(), output_dir);
    delete[] out_chain;
    delete[] out_info;
#endif

    delete[] TOED_edges;

    return 0;
}
