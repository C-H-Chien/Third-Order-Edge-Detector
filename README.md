# Third-Order Edge Detection

This is a OpenMP C++ and CUDA GPU implementation of Third-Order Edge Detection (TOED). See the referenced paper for more information. The matlab code of the paper can be found in Yuliang's [github page](https://github.com/yuliangguo/Differential_Geometry_in_Edge_Detection). <br /> 

```BibTeX
@article{kimia2018differential,
  title={Differential geometry in edge detection: accurate estimation of position, orientation and curvature},
  author={Kimia, Benjamin B and Li, Xiaoyan and Guo, Yuliang and Tamrakar, Amir},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={41},
  number={7},
  pages={1573--1586},
  year={2018},
  publisher={IEEE}
}
```

## :dependabot: Dependencies
The code has been tested in Linux-based system with the following versions of dependencies: <br /> 
(1) cuda/11.1.1 or higher <br />
(2) (Optional) OpenCV 3.X or above (only used to read an image and access image pixel values) <br />
Note that:
- CUDA version depends on the GPU. Please have it checked to use the correct cuda version, _e.g._, using the ``$ nvidia-smi`` command. <br />
- If you do not use OpenCV, set ``OPENCV_SUPPORT`` in [indices.hpp](https://github.com/C-H-Chien/Third-Order-Edge-Detector/blob/master/indices.hpp#L5) to false, and command out the include paths and library paths in the makefiles.

## :hammer_and_wrench: Setup and Run the code
There are two make files to build and compile the code: _(i)_ ``makefile.gpu_cpu`` works for all the files, including the GPU and the CPU code. _(ii)_ ``makefile.cpu`` works only for the CPU code with double precision. You can do either one of them to build and compile the code by,
```bash
$ make -f makefile.gpu_cpu  // or make -f makefile.cpu (for CPU-only version)
```
Make sure you have changed the paths for CUDA, OpenCV, etc. in the makefiles. Once everything works perfectly, proceed to execute the code by
- For GPU+CPU
```bash
$ ./TOED <name_of_input_image> <number of CPU threads> <gpu id>
```
- For CPU only:
```bash
$ ./TOED <name_of_input_image> <number of CPU threads>
```
The argument ``<name_of_input_image>`` is mandatory while the rest are optional. If OpenCV is supported, any type of images should be supported. Otherwise, only `.pgm` image file is accepted. A few sample images are provided in `./input_images/`, so you can, for example, run the code using:
```bash
$ ./TOED ./input_images/euroc_sample_img.png 4
```
You can clear out all the ``*.o`` files by
```bash
$ make -f makefile.gpu_cpu clean
```

## :tv: Display edges and orientations
After a successful run, lists of subpixel edges are written in text files named ``data_final_output_cpu.txt`` under `./output_files/`. You can use the matlab file in `./draw_edges_by_matlab/draw_edges_from_list.m` to plot the edges of the input image, or `./draw_edges_by_matlab/draw_edges_orient_from_list.m` to plot the edges and their orientations of the input image.

## :timer_clock: Some test results (To be updated)
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

## :arrow_heading_up: ChangeLogs
Jul. 23, 2023: The curvel formation code is also included in thie repo for completeness. It only accepts double precision third-order edges for now. Single precision will be made available for curvel formation in the near future.
Feb. 4, 2025: Fix a few bugs and also support OpenCV to read an image
