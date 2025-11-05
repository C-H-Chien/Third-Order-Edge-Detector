function visualize_curvelets(img, TO_edges, curvelet_list)

    figure;
    imshow(img);
    hold on;

    rng('default');
    rng(1);
    num_of_contours = size(curvelet_list, 1);
    sz = [num_of_contours 3];
    contour_RGB_color = unifrnd(0,1,sz);
    
    %> image height and width
    opts.w = size(img,2);
    opts.h = size(img,1);
    
    %> show image first
    %imshow(img);
    %hold on;
    
    max_sz_contour = size(curvelet_list, 2);
    
    %> RGB color index map
    color_indx_img_map = zeros(opts.h, opts.w);
    
    %> record all subpix positions of the contours
    all_contour_pos_x = zeros(size(curvelet_list,1), max_sz_contour);
    all_contour_pos_y = zeros(size(curvelet_list,1), max_sz_contour);
    all_contour_color = zeros(size(curvelet_list,1), 1);
    
    %> make sure that the size of RGB_color list is the same as the number of contours
    if size(curvelet_list, 1) > size(contour_RGB_color, 1)
        rng('default');
        rng(1);
        num_of_contours = size(curvelet_list, 1);
        sz = [num_of_contours 3];
        contour_RGB_color = unifrnd(0,1,sz);
    end
    
    %> loop over all contour list
    for i = 1:size(curvelet_list,1)
        contour_pos = zeros(max_sz_contour, 2);
        contour_length = 0;
        for k = 1:max_sz_contour
            if curvelet_list(i,k) > 0
                contour_length = contour_length + 1;
            end
        end
    
        for j = 1:max_sz_contour
            if curvelet_list(i,j) > 0
                px = TO_edges(curvelet_list(i,j), 1);
                py = TO_edges(curvelet_list(i,j), 2);
                if round(px) <= 0 || round(py) <= 0 || round(px) > opts.w || round(py)+10>opts.h
                   continue;
                else
                    contour_pos(j,1) = px;
                    contour_pos(j,2) = py;
                    all_contour_pos_x(i, j) = px;
                    all_contour_pos_y(i, j) = py;
                    
                    color_indx_img_map(round(py), round(px)) = i;
                    all_contour_color(i, 1) = i;
                end
            else
                break;
            end
        end
    
        %plot(contour_pos(:,1), contour_pos(:,2), '.', 'Color', contour_RGB_color(i,:), 'MarkerSize', 7);
        show_indices = find(contour_pos(:,1) > 0);
        show_contours = contour_pos(show_indices, :);
        plot(show_contours(:,1), show_contours(:,2), 'color', contour_RGB_color(i,:), 'Marker', '.', 'MarkerSize', 10);
        hold on;
    end
end
