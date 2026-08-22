clc; close all;

dst_path = '/gpfs/data/bkimia/cchien3/Third-Order-Edge-Detector/';
output_data_path = 'output_files/';

% -- read image to retrieve image height and width --
% input_img_folder = 'input_images/';
% input_img_name = 'test_undistort_left_img';
% str_readPath = strcat(dst_path, input_img_folder, input_img_name, '.jpg');

dataset_path = "/gpfs/data/bkimia/Datasets/";
dataset_name = "Middlebury_Stereo";
stereo_name = "scenes2021";
scene_name = "chess1";
img_name = "im0.png";
str_readPath = fullfile(dataset_path, dataset_name, stereo_name, "data", scene_name, img_name);
img = imread(str_readPath);
img_width = size(img,2);
img_height = size(img,1);

% -- read edge map text files --
edge_list_pts_file = 'data_final_output_cpu.txt';
full_edge_file = fullfile(dst_path, output_data_path, edge_list_pts_file);
toed_edges = importdata(full_edge_file);
toed_orient_vec = [cos(toed_edges(:,3)), sin(toed_edges(:,3))];

figure;
quiver_mag = 0.3;
imshow(img); hold on;
scatter(toed_edges(:,1)+1, toed_edges(:,2)+1, 'c.'); hold on;
quiver(toed_edges(:,1)+1, toed_edges(:,2)+1, ...
       quiver_mag*toed_orient_vec(:,1), quiver_mag*toed_orient_vec(:,2), 0, ...
       'Color', 'c', 'LineWidth', 2, 'MaxHeadSize', 1.5);
set(gcf,'color','w');