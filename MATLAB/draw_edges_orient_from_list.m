clc; close all;

dst_path = '/home/chchien/BrownU/canny/';
output_data_path = 'output_data/';

% -- read image to retrieve image height and width --
input_img_folder = 'input_images/';
input_img_name = '2018';
str_readPath = strcat(dst_path, input_img_folder, input_img_name, '.pgm');
img = imread(str_readPath);
img_width = size(img,2);
img_height = size(img,1);

% -- read edge map text files --
edge_list_pts_file = 'data_final_output_cpu.txt';
full_edge_file = fullfile(dst_path, output_data_path, edge_list_pts_file);
edge_map = fopen(full_edge_file, 'r');
ldata = textscan(edge_map, '%f\t%f\t%f\t%f\t%f\t%f\t%f', 'CollectOutput', true );
TO_edges = ldata{1,1};

% figure;
% plot(TO_edges(:,1), TO_edges(:,2),'.', 'Color', [0.1249  0.7851  0.6067]);
% xlim([-50, 350]);
% ylim([-50, 500]);
% axis equal;
% set(gcf,'color','w');

figure;
plot([TO_edges(:,1)+0.5*cos(edg(:,3)) edg(:,1)-0.5*cos(edg(:,3))], [edg(:,2)+0.5*sin(edg(:,3)) edg(:,2)-0.5*sin(edg(:,3))], 'b');
xlim([-50, 350]);
ylim([-50, 500]);
axis equal;
set(gcf,'color','w');