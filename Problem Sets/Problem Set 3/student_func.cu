/* Udacity Homework 3
HDR Tone-mapping

Background HDR
==============

A High Dynamic Range (HDR) image contains a wider variation of intensity
and color than is allowed by the RGB format with 1 byte per channel that we
have used in the previous assignment.

To store this extra information we use single precision floating point for
each channel.  This allows for an extremely wide range of intensity values.

In the image for this assignment, the inside of church with light coming in
through stained glass windows, the raw input floating point values for the
channels range from 0 to 275.  But the mean is .41 and 98% of the values are
less than 3!  This means that certain areas (the windows) are extremely bright
compared to everywhere else.  If we linearly map this [0-275] range into the
[0-255] range that we have been using then most values will be mapped to zero!
The only thing we will be able to see are the very brightest areas - the
windows - everything else will appear pitch black.

The problem is that although we have cameras capable of recording the wide
range of intensity that exists in the real world our monitors are not capable
of displaying them.  Our eyes are also quite capable of observing a much wider
range of intensities than our image formats / monitors are capable of
displaying.

Tone-mapping is a process that transforms the intensities in the image so that
the brightest values aren't nearly so far away from the mean.  That way when
we transform the values into [0-255] we can actually see the entire image.
There are many ways to perform this process and it is as much an art as a
science - there is no single "right" answer.  In this homework we will
implement one possible technique.

Background Chrominance-Luminance
================================

The RGB space that we have been using to represent images can be thought of as
one possible set of axes spanning a three dimensional space of color.  We
sometimes choose other axes to represent this space because they make certain
operations more convenient.

Another possible way of representing a color image is to separate the color
information (chromaticity) from the brightness information.  There are
multiple different methods for doing this - a common one during the analog
television days was known as Chrominance-Luminance or YUV.

We choose to represent the image in this way so that we can remap only the
intensity channel and then recombine the new intensity values with the color
information to form the final image.

Old TV signals used to be transmitted in this way so that black & white
televisions could display the luminance channel while color televisions would
display all three of the channels.


Tone-mapping
============

In this assignment we are going to transform the luminance channel (actually
the log of the luminance, but this is unimportant for the parts of the
algorithm that you will be implementing) by compressing its range to [0, 1].
To do this we need the cumulative distribution of the luminance values.

Example
-------

input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
min / max / range: 0 / 9 / 9

histo with 3 bins: [4 7 3]

cdf : [4 11 14]


Your task is to calculate this cumulative distribution by following these
steps.

*/


#include "reference_calc.cpp"
#include "utils.h"
#include "float.h"

__global__ void blelloch_scan_kernel(unsigned int *d_inout, const size_t num_bins)
{
size_t tid = threadIdx.x;
int i = 0;
for(i = 1; i < num_bins; i <<= 1){
if((tid + 1) % (i * 2) == 0){
d_inout[tid] += d_inout[tid - i];
}
__syncthreads();
}
if(tid == num_bins - 1){
d_inout[tid] = 0;
}
for(i = num_bins / 2; i >= 1; i /= 2){
if((tid + 1) % (i * 2) == 0){
unsigned int tmp = d_inout[tid - i];
d_inout[tid - i] = d_inout[tid];
d_inout[tid] += tmp;
}
__syncthreads();
}
}

__global__ void hillis_scan_kernel(unsigned int *d_inout, const size_t num_bins){
size_t tid = threadIdx.x;
extern __shared__ unsigned int sdata2[];
sdata2[tid] = d_inout[tid];
__syncthreads();
int i = 0;
for(i = 1; i < num_bins; i <<= 1){
if(tid >= i){
atomicAdd(&sdata2[tid], sdata2[tid - i]);
//sdata2[tid] += sdata2[tid - i];
}
__syncthreads();
}
// ---Inclusive to exclusive
if(tid == 0){
d_inout[tid] = 0;
}
else{
d_inout[tid] = sdata2[tid - 1];
}
}

__global__ void create_histogram_kernel(const float *d_in, unsigned int *d_out, const float minval,
const float valrange, const size_t num_bins){
size_t tid = threadIdx.x;
size_t abs_idx = blockIdx.x * blockDim.x + tid;
size_t bin = (d_in[abs_idx] - minval) / valrange * num_bins;
if(bin == num_bins){
// --- Out of range case
bin--;
}
atomicAdd(&d_out[bin], 1);
}

__global__ void shmem_reduce_kernel(const float *d_in, float *d_out, bool is_min_op){
extern __shared__ float sdata1[];
size_t tid = threadIdx.x;
size_t abs_idx = blockIdx.x * blockDim.x + tid;
sdata1[tid] = d_in[abs_idx];
// --- Before going further, we have to make sure that all the shared memory loads have been completed
__syncthreads();
// --- Reduction in shared memory. Only half of the threads contribute to reduction
size_t s = 0;
for(s = (blockDim.x >> 1); s > 0; s >>= 1){
if(tid < s){
if(is_min_op){
sdata1[tid] = fminf(sdata1[tid], sdata1[tid + s]);
}
else{
sdata1[tid] = fmaxf(sdata1[tid], sdata1[tid + s]);
}
}
// --- Make sure all min op at one stage are done
__syncthreads();
}
// --- Only thread 0 writes result for this block back to global mem
if(tid == 0){
d_out[blockIdx.x] = sdata1[0];
}
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
unsigned int* const d_cdf,
float &min_logLum,
float &max_logLum,
const size_t numRows,
const size_t numCols,
const size_t numBins)
{
/*
* 1) find the minimum and maximum value in the input logLuminance channel
*    store in min_logLum and max_logLum
*/
const size_t numSize = numRows * numCols;
const size_t tnum = (1 << 10);
size_t bnum = numSize >> 10;
float *d_tempbuf;
checkCudaErrors(cudaMalloc((void **)&d_tempbuf, bnum * sizeof(float)));
float *d_min;
float *d_max;
checkCudaErrors(cudaMalloc((void **)&d_min, 1 * sizeof(float)));
checkCudaErrors(cudaMalloc((void **)&d_max, 1 * sizeof(float)));
shmem_reduce_kernel<<<bnum, tnum, tnum * sizeof(float)>>>(d_logLuminance, d_tempbuf, true);
cudaDeviceSynchronize();
shmem_reduce_kernel<<<1, bnum, bnum * sizeof(float)>>>(d_tempbuf, d_min, true);
cudaDeviceSynchronize();
checkCudaErrors(cudaMemset(d_tempbuf, 0, bnum * sizeof(float)));
shmem_reduce_kernel<<<bnum, tnum, tnum * sizeof(float)>>>(d_logLuminance, d_tempbuf, false);
cudaDeviceSynchronize();
shmem_reduce_kernel<<<1, bnum, bnum * sizeof(float)>>>(d_tempbuf, d_max, false);
cudaDeviceSynchronize();

checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));
checkCudaErrors(cudaFree(d_tempbuf));
checkCudaErrors(cudaFree(d_min));
checkCudaErrors(cudaFree(d_max));

/*
* 2) subtract them to find the range
*/
float range_logLum = max_logLum - min_logLum;
/*
* 3) generate a histogram of all the values in the logLuminance channel using
*    the formula: bin = (lum[i] - lumMin) / lumRange * numBins
*/
checkCudaErrors(cudaMemset(d_cdf, 0, sizeof(unsigned int)* numBins));
create_histogram_kernel<<<bnum, tnum>>>(d_logLuminance, d_cdf, min_logLum,
range_logLum, numBins);
cudaDeviceSynchronize();
/*
* 4) Perform an exclusive scan (prefix sum) on the histogram to get
*    the cumulative distribution of luminance values (this should go in the
*    incoming d_cdf pointer which already has been allocated for you)
*/
//blelloch_scan_kernel<<<1, numBins, numBins * sizeof(unsigned int)>>>(d_cdf, numBins);
blelloch_scan_kernel<<<1, numBins>>>(d_cdf, numBins);
cudaDeviceSynchronize();
}
