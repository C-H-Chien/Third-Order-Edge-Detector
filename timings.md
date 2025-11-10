## :timer_clock: Some test results (To be updated)
Test image resolution: 321 x 420 <br />

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