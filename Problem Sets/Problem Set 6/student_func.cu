//Udacity HW 6
//Poisson Blending

/* Background
==========

The goal for this assignment is to take one image (the source) and
paste it into another image (the destination) attempting to match the
two images so that the pasting is non-obvious. This is
known as a "seamless clone".

The basic ideas are as follows:

1) Figure out the interior and border of the source image
2) Use the values of the border pixels in the destination image
as boundary conditions for solving a Poisson equation that tells
us how to blend the images.

No pixels from the destination except pixels on the border
are used to compute the match.

Solving the Poisson Equation
============================

There are multiple ways to solve this equation - we choose an iterative
method - specifically the Jacobi method. Iterative methods start with
a guess of the solution and then iterate to try and improve the guess
until it stops changing.  If the problem was well-suited for the method
then it will stop and where it stops will be the solution.

The Jacobi method is the simplest iterative method and converges slowly -
that is we need a lot of iterations to get to the answer, but it is the
easiest method to write.

Jacobi Iterations
=================

Our initial guess is going to be the source image itself.  This is a pretty
good guess for what the blended image will look like and it means that
we won't have to do as many iterations compared to if we had started far
from the final solution.

ImageGuess_prev (Floating point)
ImageGuess_next (Floating point)

DestinationImg
SourceImg

Follow these steps to implement one iteration:

1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
else if the neighbor in on the border then += DestinationImg[neighbor]

Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

2) Calculate the new pixel value:
float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


In this assignment we will do 800 iterations.
*/



#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"

__global__ void build_mask(const uchar4* src, unsigned char* dstmask, const size_t num_rows, const size_t num_cols){
int x = threadIdx.x + blockDim.x * blockIdx.x;
int y = threadIdx.y + blockDim.y * blockIdx.y;
if(x > num_cols || y > num_rows){
return;
}
int idx = x + y * num_cols;
if(src[idx].x == 255 && src[idx].y == 255 && src[idx].z == 255){
return;
}
else{
dstmask[idx] = 1;
}
}

__global__ void build_mask_interior_border(const unsigned char* d_in, unsigned char* d_out,
const size_t num_rows, const size_t num_cols){
int x = threadIdx.x + blockDim.x * blockIdx.x;
int y = threadIdx.y + blockDim.y * blockIdx.y;
if(x > num_cols || y > num_rows){
return;
}
int idx = x + y * num_cols;
if(x){
d_out[idx] += d_in[idx - 1];
}
if(x < (num_cols - 1)){
d_out[idx] += d_in[idx + 1];
}
if(y){
d_out[idx] += d_in[idx - num_cols];
}
if(y < (num_rows - 1)){
d_out[idx] += d_in[idx + num_cols];
}
}

__global__ void separate_channel(const uchar4* const d_src, unsigned char* d_red, unsigned char* d_green,
unsigned char* d_blue, const size_t num_rows, const size_t num_cols){
int x = threadIdx.x + blockDim.x * blockIdx.x;
int y = threadIdx.y + blockDim.y * blockIdx.y;
if(x > num_cols || y > num_rows){
return;
}
int idx = x + y * num_cols;
d_red[idx] = d_src[idx].x;
d_green[idx] = d_src[idx].y;
d_blue[idx] = d_src[idx].z;
}

__global__ void uchar_to_float_array(const unsigned char* src, float* dst, const size_t num_rows,
const size_t num_cols, const unsigned char* mask){
int x = threadIdx.x + blockDim.x * blockIdx.x;
int y = threadIdx.y + blockDim.y * blockIdx.y;
if(x > num_cols || y > num_rows){
return;
}
int idx = x + y * num_cols;
if(mask[idx] == 5){
dst[idx] = (float)src[idx];
}
else{
dst[idx] = 0.0f;
}
}

__global__ void solve_poisson(const float* src, float* dst, const unsigned char* source, const unsigned char* target,
const unsigned char* mask_interior_border, const unsigned char* mask,
const size_t num_rows, const size_t num_cols){
int x = threadIdx.x + blockDim.x * blockIdx.x;
int y = threadIdx.y + blockDim.y * blockIdx.y;
if(x > num_cols || y > num_rows){
return;
}
int idx = x + y * num_cols;
if(mask_interior_border[idx] != 5){
return;
}
int neighbor[4] = {1, -1, num_cols, -num_cols};
float sum1 = 0.0f;
float sum2 = 0.0f;
int i = 0;
for(i = 0; i < 4; i++){
int neighbor_idx = idx + neighbor[i];
if(mask_interior_border[neighbor_idx] == 5){
sum1 += src[neighbor_idx];
}
else if(mask[neighbor_idx] == 1){
sum1 += (float)target[neighbor_idx];
}
sum2 += (float)(source[idx] - source[neighbor_idx]);
}
float new_value = (sum1 + sum2) / 4.0f;
dst[idx] = min(255.0f, max(0.0f, new_value));
}

__global__ void cast_union_channels_copy_to_target(const float* red, const float* green, const float* blue,
uchar4* const dst, const size_t num_rows, const size_t num_cols,
const unsigned char* mask_interior_border){
int x = threadIdx.x + blockDim.x * blockIdx.x;
int y = threadIdx.y + blockDim.y * blockIdx.y;
if(x > num_cols || y > num_rows){
return;
}
int idx = x + y * num_cols;
if(mask_interior_border[idx] != 5){
return;
}
//Alpha should be 255 for no transparency
uchar4 interior_result = make_uchar4((unsigned char)red[idx], (unsigned char)green[idx],
(unsigned char)blue[idx], 255);
dst[idx] = interior_result;
}

void your_blend(const uchar4* const h_sourceImg,  //IN
const size_t numRowsSource, const size_t numColsSource,
const uchar4* const h_destImg, //IN
uchar4* const h_blendedImg){ //OUT
/*
1) Compute a mask of the pixels from the source image to be copied
The pixels that shouldn't be copied are completely white, they
have R=255, G=255, B=255.  Any other pixels SHOULD be copied.
*/
const dim3 num_threads(16, 32, 1);
//const dim3 num_blocks((numColsSource + 15) / 16, (numRowsSource + 31) / 32, 1);
const dim3 num_blocks(1 + numColsSource / num_threads.x, 1 + numRowsSource / num_threads.y, 1);
uchar4* d_source_img = NULL;
cudaMalloc(&d_source_img, numRowsSource * numColsSource * sizeof(uchar4));
cudaMemcpy(d_source_img, h_sourceImg, numRowsSource * numColsSource * sizeof(uchar4), cudaMemcpyHostToDevice);
unsigned char* d_source_mask = NULL;
cudaMalloc(&d_source_mask, numRowsSource * numColsSource * sizeof(unsigned char));
cudaMemset(d_source_mask, 0, numRowsSource * numColsSource * sizeof(unsigned char));
build_mask<<<num_blocks, num_threads>>>(d_source_img, d_source_mask, numRowsSource, numColsSource);
cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
/*
2) Compute the interior and border regions of the mask.  An interior
pixel has all 4 neighbors also inside the mask.  A border pixel is
in the mask itself, but has at least one neighbor that isn't.
*/
unsigned char* d_source_mask_interior_border = NULL;
cudaMalloc(&d_source_mask_interior_border, numRowsSource * numColsSource * sizeof(unsigned char));
cudaMemcpy(d_source_mask_interior_border, d_source_mask,
numRowsSource * numColsSource * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
build_mask_interior_border<<<num_blocks, num_threads>>>(d_source_mask, d_source_mask_interior_border,
numRowsSource, numColsSource);
cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
/*
3) Separate out the incoming image into three separate channels
*/
unsigned char* d_source_red = NULL;
unsigned char* d_source_green = NULL;
unsigned char* d_source_blue = NULL;
cudaMalloc(&d_source_red, numRowsSource * numColsSource * sizeof(unsigned char));
cudaMalloc(&d_source_green, numRowsSource * numColsSource * sizeof(unsigned char));
cudaMalloc(&d_source_blue, numRowsSource * numColsSource * sizeof(unsigned char));
separate_channel<<<num_blocks, num_threads>>>(d_source_img, d_source_red, d_source_green, d_source_blue,
numRowsSource, numColsSource);
cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
uchar4* d_target_img = NULL;
cudaMalloc(&d_target_img, numRowsSource * numColsSource * sizeof(uchar4));
cudaMemcpy(d_target_img, h_destImg, numRowsSource * numColsSource * sizeof(uchar4), cudaMemcpyHostToDevice);
unsigned char* d_target_red = NULL;
unsigned char* d_target_green = NULL;
unsigned char* d_target_blue = NULL;
cudaMalloc(&d_target_red, numRowsSource * numColsSource * sizeof(unsigned char));
cudaMalloc(&d_target_green, numRowsSource * numColsSource * sizeof(unsigned char));
cudaMalloc(&d_target_blue, numRowsSource * numColsSource * sizeof(unsigned char));
separate_channel<<<num_blocks, num_threads>>>(d_target_img, d_target_red, d_target_green, d_target_blue,
numRowsSource, numColsSource);
cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
/*
4) Create two float(!) buffers for each color channel that will
act as our guesses.  Initialize them to the respective color
channel of the source image since that will act as our intial guess.
*/
float* d_red_buffer_one = NULL;
float* d_red_buffer_two = NULL;
float* d_green_buffer_one = NULL;
float* d_green_buffer_two = NULL;
float* d_blue_buffer_one = NULL;
float* d_blue_buffer_two = NULL;
const int fsize = numColsSource * numRowsSource * sizeof(float);
cudaMalloc(&d_red_buffer_one, fsize);
cudaMalloc(&d_red_buffer_two, fsize);
cudaMalloc(&d_green_buffer_one, fsize);
cudaMalloc(&d_green_buffer_two, fsize);
cudaMalloc(&d_blue_buffer_one, fsize);
cudaMalloc(&d_blue_buffer_two, fsize);
uchar_to_float_array<<<num_blocks, num_threads>>>(d_source_red, d_red_buffer_one,
numRowsSource, numColsSource, d_source_mask_interior_border);
uchar_to_float_array<<<num_blocks, num_threads>>>(d_source_red, d_red_buffer_two,
numRowsSource, numColsSource, d_source_mask_interior_border);
uchar_to_float_array<<<num_blocks, num_threads>>>(d_source_green, d_green_buffer_one,
numRowsSource, numColsSource, d_source_mask_interior_border);
uchar_to_float_array<<<num_blocks, num_threads>>>(d_source_green, d_green_buffer_two,
numRowsSource, numColsSource, d_source_mask_interior_border);
uchar_to_float_array<<<num_blocks, num_threads>>>(d_source_blue, d_blue_buffer_one,
numRowsSource, numColsSource, d_source_mask_interior_border);
uchar_to_float_array<<<num_blocks, num_threads>>>(d_source_blue, d_blue_buffer_two,
numRowsSource, numColsSource, d_source_mask_interior_border);
cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
/*
5) For each color channel perform the Jacobi iteration described
above 800 times.
*/
int i = 0;
for(i = 0; i < 800; i++){
if(i % 2 == 0){
solve_poisson<<<num_blocks, num_threads>>>(d_red_buffer_one, d_red_buffer_two, d_source_red,
d_target_red, d_source_mask_interior_border, d_source_mask,
numRowsSource, numColsSource);
solve_poisson<<<num_blocks, num_threads>>>(d_green_buffer_one, d_green_buffer_two, d_source_green,
d_target_green, d_source_mask_interior_border, d_source_mask,
numRowsSource, numColsSource);
solve_poisson<<<num_blocks, num_threads>>>(d_blue_buffer_one, d_blue_buffer_two, d_source_blue,
d_target_blue, d_source_mask_interior_border, d_source_mask,
numRowsSource, numColsSource);
}
else{
solve_poisson<<<num_blocks, num_threads>>>(d_red_buffer_two, d_red_buffer_one, d_source_red,
d_target_red, d_source_mask_interior_border, d_source_mask,
numRowsSource, numColsSource);
solve_poisson<<<num_blocks, num_threads>>>(d_green_buffer_two, d_green_buffer_one, d_source_green,
d_target_green, d_source_mask_interior_border, d_source_mask,
numRowsSource, numColsSource);
solve_poisson<<<num_blocks, num_threads>>>(d_blue_buffer_two, d_blue_buffer_one, d_source_blue,
d_target_blue, d_source_mask_interior_border, d_source_mask,
numRowsSource, numColsSource);
}
cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
/*
6) Create the output image by replacing all the interior pixels
in the destination image with the result of the Jacobi iterations.
Just cast the floating point values to unsigned chars since we have
already made sure to clamp them to the correct range.
*/
cast_union_channels_copy_to_target<<<num_blocks, num_threads>>>(d_red_buffer_one, d_green_buffer_one,
d_blue_buffer_one, d_target_img,
numRowsSource, numColsSource,
d_source_mask_interior_border);
cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

cudaMemcpy(h_blendedImg, d_target_img, numRowsSource * numColsSource * sizeof(uchar4), cudaMemcpyDeviceToHost);

checkCudaErrors(cudaFree(d_source_img));
checkCudaErrors(cudaFree(d_source_mask));
checkCudaErrors(cudaFree(d_source_mask_interior_border));
checkCudaErrors(cudaFree(d_target_img));
checkCudaErrors(cudaFree(d_target_red));
checkCudaErrors(cudaFree(d_target_green));
checkCudaErrors(cudaFree(d_target_blue));
checkCudaErrors(cudaFree(d_source_red));
checkCudaErrors(cudaFree(d_source_green));
checkCudaErrors(cudaFree(d_source_blue));
checkCudaErrors(cudaFree(d_red_buffer_one));
checkCudaErrors(cudaFree(d_red_buffer_two));
checkCudaErrors(cudaFree(d_green_buffer_one));
checkCudaErrors(cudaFree(d_green_buffer_two));
checkCudaErrors(cudaFree(d_blue_buffer_one));
checkCudaErrors(cudaFree(d_blue_buffer_two));
/*
Since this is final assignment we provide little boilerplate code to
help you.  Notice that all the input/output pointers are HOST pointers.

You will have to allocate all of your own GPU memory and perform your own
memcopies to get data in and out of the GPU memory.

Remember to wrap all of your calls with checkCudaErrors() to catch any
thing that might go wrong.  After each kernel call do:

cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

to catch any errors that happened while executing the kernel.
*/



/* The reference calculation is provided below, feel free to use it
for debugging purposes.
*/

/*
uchar4* h_reference = new uchar4[srcSize];
reference_calc(h_sourceImg, numRowsSource, numColsSource,
h_destImg, h_reference);

checkResultsEps((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * srcSize, 2, .01);
delete[] h_reference; */
}

