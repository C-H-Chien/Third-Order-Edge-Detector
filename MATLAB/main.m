clc; clear; close all;

img_name = "input_images/0_colors.png";
img_ = imread(img_name);
img_ = double(rgb2gray(img_));
[img_h, img_w] = size(img_);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Third-Order Edge Detector
% outputs: [Subpixel_X Subpixel_Y Orientation Confidence]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = 1;
sigma = 1;
thresh = 1;

[TO_edges, ~, ~, ~] = third_order_edge_detector(img_, sigma, n, thresh, 1);

%> (Optional) Save a list of third-order edges as a .txt file
% output_file_path        = "/your/output/file_name.txt";
% writematrix(TO_edges, output_file_path, 'Delimiter', 'tab');
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Curvel Formation
% Outputs: (i) chain: edge id chains of local curve (curvel)
%          (ii) info: [isForward ref_pt.x() ref_pt.y() ref_theta pt.x() pt.y() theta k length property]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opts.nrad = 3.5;
opts.gap = 1.5;
opts.dx = 0.4;
opts.dt = 15;
opts.token_len = 1;
opts.max_k = 0.3;
opts.cvlet_style = 2;
opts.max_size_to_group = 4;
opts.output_type = 0;
% perform Curvature Estimation
[chain, info] = form_curvelet_mex(TO_edges, img_h, img_w,...
opts.nrad, opts.gap, opts.dx, opts.dt/180*pi, opts.token_len, opts.max_k,...
opts.cvlet_style, opts.max_size_to_group, opts.output_type);

%% 
%> An Example of super-imposing third-order edges on an image

img_ = imread(img_name);
img_ = double(rgb2gray(img_));
[TO_edges, ~, ~, ~] = third_order_edge_detector(img_, sigma, n, 3, 1);
toed_orient_vec = [cos(TO_edges(:,3)), sin(TO_edges(:,3))];

figure;
quiver_mag = 0.3;
imshow(uint8(img_)); hold on;
plot(TO_edges(:,1), TO_edges(:,2), 'co', 'MarkerSize', 4);
quiver(TO_edges(:,1), TO_edges(:,2), ...
       quiver_mag*toed_orient_vec(:,1), quiver_mag*toed_orient_vec(:,2), 0, ...
       'Color', 'c', 'LineWidth', 2, 'MaxHeadSize', 1.5);

%%
%> An Example of visualizing curvels on an image

anchor_edge_index = 10;
curvelet_index = find(chain(:,1) == 10);
target_curvelet = chain(curvelet_index, :);

visualize_curvelets(uint8(img_), TO_edges, target_curvelet);
