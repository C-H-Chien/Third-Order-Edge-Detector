

void curvelet_formation( /* OUTPUTS */
                         //double* output_info, int* output_chain,
                         //int nl, mxArray *pl[], 
                         /* INPUTS */
                         int height, int width, double* TOED_edges, int edge_num, int edge_data_sz, 
                         double nrad, double gap, double dx, double dt, double token_len, double max_k, 
                         unsigned curvelet_style, unsigned group_max_sz, unsigned out_type);