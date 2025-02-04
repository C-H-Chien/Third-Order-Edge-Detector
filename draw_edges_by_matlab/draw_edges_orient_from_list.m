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
% edge_map = fopen(full_edge_file, 'r');
% ldata = textscan(edge_map, '%f\t%f\t%f\t%f\t%f\t%f\t%f', 'CollectOutput', true );
TO_edges = importdata(full_edge_file);

% figure;
% plot(TO_edges(:,1), TO_edges(:,2),'.', 'Color', [0.1249  0.7851  0.6067]);
% xlim([-50, 350]);
% ylim([-50, 500]);
% axis equal;
% set(gcf,'color','w');

figure;
imshow(img); hold on;
for i = 1:size(TO_edges, 1)
    plot([TO_edges(:,1)+0.5*cos(TO_edges(:,3)) TO_edges(:,1)-0.5*cos(TO_edges(:,3))], [TO_edges(:,2)+0.5*sin(TO_edges(:,3)) TO_edges(:,2)-0.5*sin(TO_edges(:,3))], 'c');
    hold on;
end
set(gcf,'color','w');