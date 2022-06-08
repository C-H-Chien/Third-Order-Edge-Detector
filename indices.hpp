#ifndef INDICES_HPP
#define INDICES_HPP
// macros for flexible axis

// Third-Order Edge Extraction Enabler
#define ThirdOrderEdgeDetector              (1)

// Write Data to File Enabler
#define WriteDataToFile                     (0)

// cpu
#define img(i,j)                        img[(i) * img_width + (j)]
#define Ix(i, j)                            Ix[(i) * interp_img_width + (j) ]
#define Iy(i, j)                            Iy[(i) * interp_img_width + (j) ]
#define I_grad_mag(i, j)            I_grad_mag[(i) * interp_img_width + (j) ]
#define I_orient(i, j)                I_orient[(i) * interp_img_width + (j) ]

#define subpix_pos_x_map(i,j)                 subpix_pos_x_map[(i) * interp_img_width + (j)]
#define subpix_pos_y_map(i,j)                 subpix_pos_y_map[(i) * interp_img_width + (j)]
#define subpix_edge_pts_final(i,j)       subpix_edge_pts_final[(i) * num_of_edge_data + (j)]

// gpu
#define dev_img(i,j)                           dev_img[(i) * img_width + (j)]
#define dev_Ix(i,j)                             dev_Ix[(i) * interp_img_width + (j)]
#define dev_Iy(i,j)                             dev_Iy[(i) * interp_img_width + (j)]
#define dev_I_grad_mag(i, j)            dev_I_grad_mag[(i) * interp_img_width + (j)]
#define dev_I_orient(i, j)                dev_I_orient[(i) * interp_img_width + (j)]

#define dev_subpix_pos_x_map(i,j)         dev_subpix_pos_x_map[(i) * interp_img_width + (j)]
#define dev_subpix_pos_y_map(i,j)         dev_subpix_pos_y_map[(i) * interp_img_width + (j)]



#endif // INDICES_HPP