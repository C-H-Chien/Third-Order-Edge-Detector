# Third-Order Edge Detection

This is a OpenMP C++ and CUDA GPU implementation of Third-Order Edge Detection (TOED) from the paper: <br /> 
``Kimia, Benjamin B., Xiaoyan Li, Yuliang Guo, and Amir Tamrakar. "Differential geometry in edge detection: accurate estimation of position, orientation and curvature." IEEE transactions on pattern analysis and machine intelligence 41, no. 7 (2018): 1573-1586.`` <br />

The matlab code of the paper can be found in Yuliang's [github page](https://github.com/yuliangguo/Differential_Geometry_in_Edge_Detection).

## 0. Updates
Jul. 23, 2023: The curvel formation code is also included in thie repo for completeness. It only accepts double precision third-order edges for now. Single precision will be made available for curvel formation in the near future.

## 1. Dependencies
The code has been tested in Linux-based system with the following versions of dependencies: <br /> 
(1) g++ version 10.2 or higher <br />
(2) cuda/11.1.1 or higher <br />
Other system or lower version of gcc might work but not tested yet. Cuda version depends on the GPU. Please have it checked to use the correct cuda version.

## 2. Run the code
Under the repo directory, simply do
```bash
make toed
```
Then, run the execution with input arguments
```bash
./TOED <name_of_input_image> <number of CPU threads> <gpu id>
```
The argument ``<name_of_input_image>`` is mandatory while the other two are optional. So far only `.pgm` image file is accepted. One sample image is provided under folder `./input_images/`, so you can, for example, run the code using:
```bash
./TOED ./input_images/2018.pgm 4
```

## 3. Display edges and orientations
After a successful run, lists of subpixel edges are written in text files named ``data_final_output_cpu.txt`` under `./test_files/`. You can use the matlab file in `./draw_edges_by_matlab/draw_edges_from_list.m` to plot the edges of the input image, or `./draw_edges_by_matlab/draw_edges_orient_from_list.m` to plot the edges and its orientations of the input image.

## 4. Some test results
CPU: Intel(R) Xeon(R) Gold 6242 CPU @ 2.80GHz <br />
GPU: NVIDIA QuadroRTX 6000 <br />
**Double Precision Test**: <br />
 ==> CPU Test (OpenMP 4 threads)  <br /> 
============================================ <br />
- Time of image convolution (OpenMP): 261.332 (ms) <br />
- Time of NMS (OpenMP): 2.62396 (ms) <br /> <br />

 ==> GPU Test  <br />
============================================= <br />
- GPU Convolution time =  16.5035 ms <br />
- GPU NMS time =   0.2988 ms <br /> <br />

**Single Precision Test**: <br />
 ==> CPU Test (OpenMP 4 threads) <br />
============================================= <br />
- Time of image convolution (OpenMP): 257.492 (ms) <br />
- Time of NMS (OpenMP): 2.32577 (ms) <br /> <br />

 ==> GPU Test  <br />
============================================= <br />
- GPU Convolution time =   0.7848 ms <br />
- GPU NMS time =   0.2951 ms <br />

Note that the above timings could change according to the input image size, the power of CPU/GPU.
