function save_edg(filename, edg, dim)
    %> Save third-order edges as an `.edg` file

    fid = fopen(filename, 'w');

    %> Output header
    fprintf(fid, '# EDGE_MAP v3.0\n');
    fprintf(fid, '\n');
    fprintf(fid, '# Format :  [Pixel_Pos]  Pixel_Dir Pixel_Conf  [Sub_Pixel_Pos] Sub_Pixel_Dir Sub_Pixel_Conf Sub_Pixel_Conf \n');
    fprintf(fid, '\n');

    %> Write out width and height info in the header
    fprintf(fid, 'WIDTH=%d \n', dim(1));
    fprintf(fid, 'HEIGHT=%d \n', dim(2));

    %> Write out edge count
    edge_cnt = size(edg, 1);
    fprintf(fid, 'EDGE_COUNT=%d \n', edge_cnt);
    fprintf(fid, '\n\n');

    %> Write each edge into the file
    for i = 1:edge_cnt
        fprintf(fid, '[%d, %d]    %f %f  [%f, %f]  %f %f 0 \n', round(edg(i,1)), round(edg(i,2)), edg(i,3), edg(i,4), edg(i,1), edg(i,2), edg(i,3), edg(i,4));
    end

    %> Close the file
    fclose(fid);
end