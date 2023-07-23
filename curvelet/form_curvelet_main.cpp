#ifndef FORM_CURVELET_MAIN_CPP
#define FORM_CURVELET_MAIN_CPP

/*****************************************************************************
// file: form_curvelet_mex.cxx
// author: Xiaoyan Li
// date: 01/19/2015
//       An algorithm to form curvelet from input edgemap
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
    //edgeinfo._data = (double*) mxGetData(pr[0]);
    edgeinfo._data = TOED_edges;
    
    //int h = (int) mxGetM(pr[0]); 
    int h = edge_num; 
    edgeinfo.set_h(h);
    //int w = (int) mxGetN(pr[0]); 
    int w = edge_data_sz; 
    edgeinfo.set_w(w);

    //unsigned output_type = mxGetScalar(pr[11]);
    unsigned output_type = out_type;
    
//    mexPrintf("inputs are passed to c\n");
    
    // assign initial settings
    //unsigned cvlet_type = mxGetScalar(pr[9]);
    unsigned cvlet_type = curvelet_style;
    unsigned max_size_to_group = group_max_sz;
    bool bCentered_grouping = cvlet_type==0 || cvlet_type==1, bBidirectional_grouping = cvlet_type==0 || cvlet_type==2;
    
    // constructor of a set of classes
    /*form_curvelet_process curvelet_pro(edgeinfo, unsigned(height)),unsigned(width)),
                                       double(mxGetScalar(pr[3])), double(mxGetScalar(pr[4])),
                                       double(mxGetScalar(pr[5])), double(mxGetScalar(pr[6])),
                                       double(mxGetScalar(pr[7])), double(mxGetScalar(pr[8])),
                                       max_size_to_group,
                                       bCentered_grouping, bBidirectional_grouping);*/
    
    form_curvelet_process curvelet_pro(edgeinfo, unsigned(height),unsigned(width),
                                       nrad, gap,
                                       dx, dt,
                                       token_len, max_k,
                                       max_size_to_group,
                                       bCentered_grouping, bBidirectional_grouping);
//    mexPrintf("process is constructed\n");
    
    curvelet_pro.execute();
//    mexPrintf("process is executed\n");
    
    // create memory for output
    unsigned out_h,out_w, info_w;
    
    curvelet_pro.get_output_size(out_h,out_w,output_type);
    //const mwSize ds1[2] = {mwSize(outh), mwSize(outw)};
    //const unsigned ds1[2] = {out_h, out_w};

    int *out_chain = new int[out_h * out_w];

    if(output_type==0)
        info_w = 10;
    else if(output_type==1)
        info_w = 1;
    else
        info_w = 12;
    //const mwSize ds2[2] = {mwSize(outh), mwSize(infow)};
    //const unsigned ds2[2] = {out_h, info_w};

    double *out_info = new double[out_h * info_w];

    arrayi chain;
    //pl[0] = mxCreateNumericArray(2,ds1,mxINT32_CLASS, mxREAL);
    //chain._data = (int*) mxGetData(pl[0]);
    chain._data = out_chain;
    chain.set_h(out_h);
    chain.set_w(out_w);
    arrayd info;
    if (output_type!=1) {
        //pl[1] = mxCreateNumericArray(2,ds2,mxDOUBLE_CLASS,mxREAL);

        //info._data = (double*) mxGetData(pl[1]);
        info._data = out_info;
    }
    info.set_h(out_h);
    info.set_w(info_w);
//        mexPrintf("output memories are created\n");
    
    curvelet_pro.get_output_arrary( chain, info, output_type );
//        mexPrintf("outputs are assigned\n");

    delete[] out_info;
    delete[] out_chain;

}
/*
void mexFunction( int nl, mxArray *pl[], int nr, const mxArray *pr[] )
{
    // Let time how long this takes
    // Start timer
    // std::clock_t t;
    
    // check and get inputs
    if(nr != 12) mexErrMsgTxt("Twelve inputs required.");
    if(nl > 2) mexErrMsgTxt("At most two outputs expected.");
    unsigned output_type = mxGetScalar(pr[11]);
    if(nl > 1 && output_type==1) mexErrMsgTxt("At most one outputs expected for curve fragment graph output.");
    if(mxGetClassID(pr[0])!=mxDOUBLE_CLASS) mexErrMsgTxt("edgeinfo must be a double*");

    // construct and assign the subpixel edge list
    arrayd edgeinfo; 
    edgeinfo._data = (double*) mxGetData(pr[0]);
    
    int h = (int) mxGetM(pr[0]); 
    edgeinfo.set_h(h);
    int w = (int) mxGetN(pr[0]); 
    edgeinfo.set_w(w);
    
//    mexPrintf("inputs are passed to c\n");
    
    // assign initial settings
    unsigned cvlet_type = mxGetScalar(pr[9]);
    unsigned max_size_to_group = unsigned(mxGetScalar(pr[10]));
    bool bCentered_grouping = cvlet_type==0 || cvlet_type==1, bBidirectional_grouping = cvlet_type==0 || cvlet_type==2;
    
    // constructor of a set of classes
    form_curvelet_process curvelet_pro(edgeinfo, unsigned(mxGetScalar(pr[1])),unsigned(mxGetScalar(pr[2])),
                                       double(mxGetScalar(pr[3])), double(mxGetScalar(pr[4])),
                                       double(mxGetScalar(pr[5])), double(mxGetScalar(pr[6])),
                                       double(mxGetScalar(pr[7])), double(mxGetScalar(pr[8])),
                                       max_size_to_group,
                                       bCentered_grouping, bBidirectional_grouping);
//    mexPrintf("process is constructed\n");
    
    curvelet_pro.execute();
//    mexPrintf("process is executed\n");
    
    // create memory for output
    unsigned outh,outw, infow;
    curvelet_pro.get_output_size(outh,outw,output_type);
    const mwSize ds1[2] = {mwSize(outh), mwSize(outw)};
    if(output_type==0)
        infow = 10;
    else if(output_type==1)
        infow = 1;
    else
        infow = 12;
    const mwSize ds2[2] = {mwSize(outh), mwSize(infow)};
    if( nl>0 ) {
        arrayi chain;
        pl[0] = mxCreateNumericArray(2,ds1,mxINT32_CLASS, mxREAL);
        chain._data = (int*) mxGetData(pl[0]);
        chain.set_h(outh);
        chain.set_w(outw);
        arrayd info;
        if (output_type!=1 && nl>1) {
            pl[1] = mxCreateNumericArray(2,ds2,mxDOUBLE_CLASS,mxREAL);
            info._data = (double*) mxGetData(pl[1]);
        }
        info.set_h(outh);
        info.set_w(infow);
//        mexPrintf("output memories are created\n");
        
        curvelet_pro.get_output_arrary( chain, info, output_type );
//        mexPrintf("outputs are assigned\n");
    }
}
*/




#endif //FORM_CURVELET_MAIN_CPP
