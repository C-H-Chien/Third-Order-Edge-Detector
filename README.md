# Third-Order Edge Detection

This is OpenMP C++, CUDA GPU, and MATLAB implementations of Third-Order Edge Detection (TOED). See the referenced paper for more information. The original matlab code of the paper can be found in Yuliang's [github page](https://github.com/yuliangguo/Differential_Geometry_in_Edge_Detection), but here it is slightly reorganized to make it new-user friendly. <br /> 

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
## C++ and CUDA Code
### :dependabot: Dependencies
The code has been tested in Linux-based system with the following versions of dependencies: <br /> 
(1) cuda/11.1.1 or higher, if the GPU code is used <br />
(2) (Optional) OpenCV 3.X or above (only used to read an image and access image pixel values) <br />
Note that:
- CUDA version depends on the GPU. Please have it checked to use the correct cuda version, _e.g._, using the ``$ nvidia-smi`` command. <br />
- If you do not use OpenCV, set ``OPENCV_SUPPORT`` in [indices.hpp](https://github.com/C-H-Chien/Third-Order-Edge-Detector/blob/master/indices.hpp#L5) to false, and command out the include paths and library paths in the makefiles.

### :hammer_and_wrench: Setup and Run the code
There are two make files to build and compile the code: _(i)_ ``makefile.gpu_cpu`` works for all the files, including the GPU and the CPU code. _(ii)_ ``makefile.cpu`` works only for the CPU code with double precision. You can do either one of them to build and compile the code by,
```bash
$ make -f makefile.gpu_cpu  // or make -f makefile.cpu (for CPU-only version)
```
Make sure you have changed the paths for CUDA, OpenCV, etc. in the makefiles. Once everything works perfectly, proceed to execute the code by
- For GPU+CPU
```bash
$ ./TOED <name_of_input_image> <number of CPU threads> <gpu id> <output_dir>
```
- For CPU only:
```bash
$ ./TOED <name_of_input_image> <number of CPU threads> <output_dir>
```
The argument ``<name_of_input_image>`` is mandatory while the rest are optional (default output directory is ``./output_files``). If OpenCV is supported, any type of images should be supported. Otherwise, only `.pgm` image file is accepted. A few sample images are provided in `./input_images/`, so you can, for example, run the code using:
```bash
$ ./TOED ./input_images/euroc_sample_img.png 4 0 ./output_files/
```
You can clear out all the ``*.o`` files by
```bash
$ make clean
```
Note that there is also a curvelet construction code following the third-order edge detection, which is by default turned off (see the setting `CurvelFormation` in `indices.hpp`).

### :warning: Building GPU code is GPU architecture sensitive
On running GPU code, if you have built the code successfully but at some points you switch to a different GPU, the GPU kernels may be built for different GPU architecture. In this case, force a clean rebuild:
```bash
make -f makefile.gpu_cpu clean
make -f makefile.gpu_cpu
```
You should see `nvcc` recompile the `.cu` files.

### :tv: Display edges and orientations
After a successful run, lists of subpixel edges are written in text files named ``data_final_output_cpu.txt`` if running the CPU version, or `data_final_output_gpu` if running the GPU version, under the specified output directory (default ``./output_files/``). You can use the matlab file in `./MATLAB/draw_edges_from_list.m` to plot the edges of the input image.

## Subpixel Third-Order Correction on a Given Edge Map
This is a standalone CPU/OpenMP tool that applies subpixel third-order (TO) correction to an existing dense edge map (represented by a probabilitic distribution of edges on an entire image), rather than detecting edges from a raw image. Basically, the process starts with a non-maximum suppression along the given orientation, then third-order orientation correction from Gaussian derivatives of the edge map. If initial orientation is not given, a dense orientation map from the edge map is obtained by using Dollar’s Hessian formula (see [Dollar's paper](https://ieeexplore.ieee.org/abstract/document/6975234)).

### Setup and run
The code for subpixel TO correction resides in `subpix_TO_correction/`.
```bash
$ cd subpix_TO_correction
$ make                         # or: make -C subpix_TO_correction  from the repo root
$ ./TO_correct <edgemap> [orientation] [thresh] [sigma] [nthreads] [output_dir]
```
Update the OpenCV path in ``subpix_TO_correction/makefile`` if needed. Arguments after ``<edgemap>`` are optional. See `main_subpix_to_correction.cpp` for more information. 

For the inputs, 8-bit images are treated as probabilities in ``[0, 1]`` (divided by 255). For raw floating-point maps, pass a ``.txt`` matrix. Example:
```bash
$ ./TO_correct E.png O.txt 0.1 2 4 ./output_files
```

### Output
Edgels are written as 0-based pixel coordinates in the same 4-column layout as ``TOED_edges.txt`` (``x``, ``y``, orientation, strength)
To compare with MATLAB, add 1 to ``x`` and ``y``. The same MATLAB script ``./MATLAB/draw_edges_from_list.m`` enables plotting the edges from ``TO_corrected_edges.txt``.

## MATLAB Code
The MATLAB code resides in the ``MATLAB`` folder. The ``main.m`` code contains both the third-order edge detection and curvelet (curvel) extraction, with additional example code for visualization. 

## Timings
Some test results can be found in `timings.md`. 
