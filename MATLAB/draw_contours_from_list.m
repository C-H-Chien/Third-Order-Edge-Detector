clc; close all;

dst_path = '/home/chchien/BrownU/canny/';
output_data_path = 'draw_edges_by_matlab/';

% -- read image to retrieve image height and width --
input_img_folder = 'input_images/';
input_img_name = '2018';
str_readPath = strcat(dst_path, input_img_folder, input_img_name, '.pgm');
img = imread(str_readPath);
img_width = size(img,2);
img_height = size(img,1);

% -- read contour list file --
contour_list_file = 'data_subpix_contour_output.txt';
full_contour_file = fullfile(dst_path, output_data_path, contour_list_file);
contour_map = fopen(full_contour_file, 'r');
ldata = textscan(contour_map, '%f\t%f\t%f', 'CollectOutput', true );
subpix_contour_label = ldata{1,1};

% -- fetch the number of contours --
num_of_contours = max(subpix_contour_label(:,3));

% -- create a contour list with corresponding color sampled from a uniform
% distribution --
sz = [num_of_contours 3];
contour_RGB_color = unifrnd(0,1,sz);

figure;
for i = 1:size(subpix_contour_label, 1)
    contour_label = subpix_contour_label(i,3);
%     if ~isinteger(contour_label)
%         wr_str = strcat(string(i), ',', string(contour_label), '\n');
%         fprintf(wr_str);
%     end
    plot(subpix_contour_label(i,1),subpix_contour_label(i,2),'.', 'Color', contour_RGB_color(contour_label,:));
    hold on;
end
hold off;
xlim([-50, 350]);
ylim([-50, 500]);
axis equal;
set(gcf,'color','w');