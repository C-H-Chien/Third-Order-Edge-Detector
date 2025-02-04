clc; close all;

dst_path = '/gpfs/data/bkimia/cchien3/Third-Order-Edge-Detector/';
output_data_path = 'output_files/';

% -- read image to retrieve image height and width --
input_img_folder = 'input_images/';
input_img_name = 'euroc_sample_img';
str_readPath = strcat(dst_path, input_img_folder, input_img_name, '.png');
img = imread(str_readPath);
img_width = size(img,2);
img_height = size(img,1);

% -- read edge map text files --
edge_list_pts_file = 'data_final_output_cpu.txt';
full_edge_file = fullfile(dst_path, output_data_path, edge_list_pts_file);
toed_edges = importdata(full_edge_file);

% % -- strong edges --
% edge_strong_pts_file = 'data_strong.txt';
% full_strong_edge_file = fullfile(dst_path, output_data_path, edge_strong_pts_file);
% strong_edge_map = fopen(full_strong_edge_file, 'r');
% ldata_strong = textscan(strong_edge_map, '%f\t%f\t%f\t%f\t%f\t%f\t%f', 'CollectOutput', true );
% pixel2d_strong = ldata_strong{1,1};
% 
% % -- weak edges --
% edge_weak_pts_file = 'data_weak.txt';
% full_weak_edge_file = fullfile(dst_path, output_data_path, edge_weak_pts_file);
% weak_edge_map = fopen(full_weak_edge_file, 'r');
% ldata_weak = textscan(weak_edge_map, '%f\t%f\t%f\t%f\t%f\t%f\t%f', 'CollectOutput', true );
% pixel2d_weak = ldata_weak{1,1};

figure;
imshow(img); hold on;
scatter(toed_edges(:,1), toed_edges(:,2), 'c.');
set(gcf,'color','w');

% figure;
% plot(pixel2d_strong(:,1),pixel2d_strong(:,2),'.', 'Color', [0.1249  0.7851  0.6067]);
% hold on;
% plot(pixel2d_weak(:,1),pixel2d_weak(:,2),'r.');
% xlim([-50, 350]);
% ylim([-50, 500]);
% axis equal;
% set(gcf,'color','w');