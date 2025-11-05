#ifndef FORM_CURVELET_MAIN_CPP
#define FORM_CURVELET_MAIN_CPP

/*****************************************************************************
// file: form_curvelet_main.cpp
// author: Chiang-Heng Chien
// date: Jul. 23, 2023
//       An algorithm to form curvelet from input edgemap. Edited from a matlab mex function interface.
******************************************************************************/

// mex  form_curvelet_mex.cxx CC_curve_model_3d.cxx curvelet.cxx
//       curveletmap.cxx form_curvelet_process.cxx

//#include "mex.h"
#include <ctime>
#include "Array.hpp"
#include "form_curvelet_process.hpp"
#include "curvelet_utils.hpp"

/*************************************************************
Usage: 
 [chain, info] = form_curvelet_mex(edgeinfo, nrows, ncols,...
 rad, gap, dx, dt, token_len, max_k, cvlet_type, ...
 max_size_to_goup, output_type)
Input:
        edgeinfo: nx4 array storing the position, orientation
 and magnitude information;
        nrows: height of the image;
        ncols: width of the image;
        rad: radius of the grouping neighborhood around each
 edgel;
        gap: distance between two consecutive edges in a link;
        dx: position uncertainty at the reference edgel;
        dt: orientation uncertainty at the reference edgel;
        token_len:
        max_k: maximum curvature of curves in the curve bundle;
        cvlet_type: if 0, form regular anchor centered curvelet;
 if 1, anchor centered bidirectional; if 2, anchor leading bidirectional;
 if 3, ENO style (anchor leading or trailing but in the same direction).
        max_size_to_goup: the maximum numbers of edges to group;
        output_type: if 0, out put curvelet map. if 1, out put
 the curve fragment map. if 2, out put the poly arc map.
 *************************************************************/

void curvelet_formation( /* OUTPUTS */
                         //double* output_info, int* output_chain,
                         //int nl, mxArray *pl[], 
                         arrayi &chain, arrayd &info,
                         /* INPUTS */
                         int height, int width, double* TOED_edges, int edge_num, int edge_data_sz, 
                         double nrad, double gap, double dx, double dt, double token_len, double max_k, 
                         unsigned curvelet_style, unsigned group_max_sz, unsigned out_type)
{
    // pr[0] = TOED_edges
    /*
    opts.nrad = 3.5;
    opts.gap = 1.5;
    opts.dx = 0.4;
    opts.dt = 15;
    opts.token_len = 1;
    opts.max_k = 0.3;
    opts.cvlet_style = 3;
    opts.max_size_to_group = 7;
    opts.output_type = 0;
    % perform Curvature Estimation
    [chain, info] = form_curvelet_mex(TO_edges, size(img,1), size(img,2),...
    opts.nrad, opts.gap, opts.dx, opts.dt/180*pi, opts.token_len, opts.max_k,...
    opts.cvlet_style, opts.max_size_to_group, opts.output_type);
    */

    // construct and assign the subpixel edge list
    arrayd edgeinfo; 
    edgeinfo._data = TOED_edges;
    
    int h = edge_num; 
    edgeinfo.set_h(h);
    int w = edge_data_sz; 
    edgeinfo.set_w(w);

    unsigned output_type = out_type;
    
    // assign initial settings
    //unsigned cvlet_type = mxGetScalar(pr[9]);
    unsigned cvlet_type = curvelet_style;
    unsigned max_size_to_group = group_max_sz;
    bool bCentered_grouping = cvlet_type==0 || cvlet_type==1, bBidirectional_grouping = cvlet_type==0 || cvlet_type==2;
    
    // constructor of a set of classes    
    form_curvelet_process curvelet_pro(edgeinfo, unsigned(height),unsigned(width),
                                       nrad, gap,
                                       dx, dt,
                                       token_len, max_k,
                                       max_size_to_group,
                                       bCentered_grouping, bBidirectional_grouping);
    
    curvelet_pro.execute();
    
    // create memory for output
    unsigned out_h,out_w, info_w;
    
    curvelet_pro.get_output_size(out_h, out_w, output_type);

    int *out_chain = new int[out_h * out_w];

    if(output_type==0)
        info_w = 10;
    else if(output_type==1)
        info_w = 1;
    else
        info_w = 12;

    double *out_info = new double[out_h * info_w];

    arrayi local_chain;
    local_chain._data = out_chain;
    local_chain.set_h(out_h);
    local_chain.set_w(out_w);
    arrayd local_info;
    if (output_type!=1) {
        local_info._data = out_info;
    }
    local_info.set_h(out_h);
    local_info.set_w(info_w);
    
    curvelet_pro.get_output_arrary( local_chain, local_info, output_type );

    // Copy local arrays to output parameters
    chain = local_chain;
    info = local_info;

    std::cout << "info width and height: " << info.h() << ", " << info.w() << std::endl;


    delete[] out_info;
    delete[] out_chain;
}




#endif //FORM_CURVELET_MAIN_CPP
