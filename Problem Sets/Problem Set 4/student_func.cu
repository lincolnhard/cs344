//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"

/* Red Eye Removal
===============

For this assignment we are implementing red eye removal.  This is
accomplished by first creating a score for every pixel that tells us how
likely it is to be a red eye pixel.  We have already done this for you - you
are receiving the scores and need to sort them in ascending order so that we
know which pixels to alter to remove the red eye.

Note: ascending order == smallest to largest

Each score is associated with a position, when you sort the scores, you must
also move the positions accordingly.

Implementing Parallel Radix Sort with CUDA
==========================================

The basic idea is to construct a histogram on each pass of how many of each
"digit" there are.   Then we scan this histogram so that we know where to put
the output of each digit.  For example, the first 1 must come after all the
0s so we have to know how many 0s there are to be able to start moving 1s
into the correct position.

1) Histogram of the number of occurrences of each digit
2) Exclusive Prefix Sum of Histogram
3) Determine relative offset of each digit
For example [0 0 1 1 0 0 1]
->  [0 1 0 1 2 3 2]
4) Combine the results of steps 2 & 3 to determine the final
output location for each element and move it there

LSB Radix sort is an out-of-place sort and you will need to ping-pong values
between the input and output buffers we have provided.  Make sure the final
sorted results end up in the output buffer!  Hint: You may need to do a copy
at the end.

*/

__global__ void exclusive_scan(unsigned int* const d_scaninput, const size_t num_elems, unsigned int* const d_block_sum)
{
const unsigned int tid = threadIdx.x;
const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
extern __shared__ unsigned int s_block[];
if(id < num_elems)
{
s_block[tid] = d_scaninput[id];
}
else
{
s_block[tid] = 0;
}
__syncthreads();
//reduce
unsigned int i;
for(i = 2; i <= blockDim.x; i <<= 1)
{
if((tid + 1) % i == 0)
{
unsigned int neighbor_offset = i >> 1;
s_block[tid] += s_block[tid - neighbor_offset];
}
__syncthreads();
}
i >>= 1; //reset i to last value before for loop exited
//reset last element(sum of whole block) to zero
if(tid == (blockDim.x - 1))
{
d_block_sum[blockIdx.x] = s_block[tid];
s_block[tid] = 0;
}
__syncthreads();
//downsweep
for(i = i; i >= 2; i >>= 1)
{
if((tid + 1) % i == 0)
{
unsigned int neighbor_offset = i >> 1;
unsigned int temp = s_block[tid - neighbor_offset];
s_block[tid - neighbor_offset] = s_block[tid];
s_block[tid] += temp;
}
__syncthreads();
}
// copy result to global memory
if(id < num_elems)
{
d_scaninput[id] = s_block[tid];
}
}

__global__ void block_sum_add(unsigned int* const d_predicate_scan, unsigned int* const d_block_sum,
const size_t num_elems)
{
const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
if(id >= num_elems)
{
return;
}
d_predicate_scan[id] += d_block_sum[blockIdx.x];
}

__global__ void flip_predicate_check(unsigned int* const d_predicate, const size_t num_elems)
{
const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
if(id >= num_elems)
{
return;
}
d_predicate[id] = (d_predicate[id] + 1) % 2;
}

__global__ void predicate_check(const unsigned int* d_input_values, const unsigned int current_bit,
const size_t num_elems, unsigned int* d_predicate)
{
const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
if(id >= num_elems)
{
return;
}
int p = ((d_input_values[id] & current_bit) == 0);
d_predicate[id] = p;
}

__global__ void scatter(unsigned int* d_src, unsigned int* d_dst, unsigned int* d_predicate_true_scan,
unsigned int* d_predicate_false_scan, unsigned int* d_predicate_false,
unsigned int* d_predicate_true_total_sum, const size_t num_elems)
{
const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
if(id >= num_elems)
{
return;
}
unsigned int newloc = 0;
if(d_predicate_false[id] == 1)
{
newloc = d_predicate_false_scan[id] + *d_predicate_true_total_sum;
}
else
{
newloc = d_predicate_true_scan[id];
}
if(newloc < num_elems)
{
d_dst[newloc] = d_src[id];
}
}

void your_sort(unsigned int* const d_inputVals,
unsigned int* const d_inputPos,
unsigned int* const d_outputVals,
unsigned int* const d_outputPos,
const size_t numElems)
{
const unsigned int blocksize = 1024;
const unsigned int gridsize = (numElems + blocksize - 1) / blocksize;
unsigned int size = sizeof(unsigned int) * numElems;

unsigned int* d_predicate; //true: bit is zero, false: bit is one
unsigned int* d_predicate_true_scan;
unsigned int* d_predicate_false_scan;
unsigned int* d_block_sum;
unsigned int* d_predicate_true_total_sum;
unsigned int* d_predicate_false_total_sum;
cudaMalloc((void**)&d_predicate, size);
cudaMalloc((void**)&d_predicate_true_scan, size);
cudaMalloc((void**)&d_predicate_false_scan, size);
cudaMalloc((void**)&d_block_sum, gridsize * sizeof(unsigned int));
cudaMalloc((void**)&d_predicate_true_total_sum, sizeof(unsigned int));
cudaMalloc((void**)&d_predicate_false_total_sum, sizeof(unsigned int));

unsigned int bit = 0;
const unsigned int uintbits = CHAR_BIT * sizeof(unsigned int) - 1;
unsigned int currentbit = 1;
for(bit = 0; bit < uintbits; bit++) //need to run odd time, because here using d_outputXXXX as buffer
{
currentbit = (1 << bit);
if((bit + 1) % 2 == 1)//if((i & 1) == 0)
{
//in -> out
predicate_check<<<gridsize, blocksize>>>(d_inputVals, currentbit, numElems, d_predicate);
}
else
{
//out -> in
predicate_check<<<gridsize, blocksize>>>(d_outputVals, currentbit, numElems, d_predicate);
}
cudaDeviceSynchronize();

cudaMemcpy(d_predicate_true_scan, d_predicate, size,
cudaMemcpyDeviceToDevice);
cudaMemset(d_block_sum, 0, gridsize * sizeof(unsigned int));
exclusive_scan<<<gridsize, blocksize, blocksize * sizeof(unsigned int)>>>
(d_predicate_true_scan, numElems, d_block_sum);
cudaDeviceSynchronize();
exclusive_scan<<<1, blocksize, blocksize * sizeof(unsigned int)>>>
(d_block_sum, gridsize, d_predicate_true_total_sum);
cudaDeviceSynchronize();
block_sum_add<<<gridsize, blocksize>>>(d_predicate_true_scan, d_block_sum, numElems);
cudaDeviceSynchronize();
// transform predicateTrue -> predicateFalse
flip_predicate_check<<<gridsize, blocksize>>>(d_predicate, numElems);
cudaMemcpy(d_predicate_false_scan, d_predicate, size,
cudaMemcpyDeviceToDevice);
cudaMemset(d_block_sum, 0, gridsize * sizeof(unsigned int));
exclusive_scan<<<gridsize, blocksize, blocksize * sizeof(unsigned int)>>>
(d_predicate_false_scan, numElems, d_block_sum);
cudaDeviceSynchronize();
exclusive_scan<<<1, blocksize, blocksize * sizeof(unsigned int)>>>
(d_block_sum, gridsize, d_predicate_false_total_sum);
cudaDeviceSynchronize();
block_sum_add<<<gridsize, blocksize>>>(d_predicate_false_scan, d_block_sum, numElems);
cudaDeviceSynchronize();
//scatter values
if((bit + 1) % 2 == 1)//if((i & 1) == 0)
{
scatter<<<gridsize, blocksize>>>
(d_inputVals, d_outputVals, d_predicate_true_scan, d_predicate_false_scan,
d_predicate, d_predicate_true_total_sum, numElems);
cudaDeviceSynchronize();
scatter<<<gridsize, blocksize>>>
(d_inputPos, d_outputPos, d_predicate_true_scan, d_predicate_false_scan,
d_predicate, d_predicate_true_total_sum, numElems);
cudaDeviceSynchronize();
}
else
{
scatter<<<gridsize, blocksize>>>
(d_outputVals, d_inputVals, d_predicate_true_scan, d_predicate_false_scan,
d_predicate, d_predicate_true_total_sum, numElems);
cudaDeviceSynchronize();
scatter<<<gridsize, blocksize>>>
(d_outputPos, d_inputPos, d_predicate_true_scan, d_predicate_false_scan,
d_predicate, d_predicate_true_total_sum, numElems);
cudaDeviceSynchronize();
}
}
cudaFree(d_predicate);
cudaFree(d_predicate_true_scan);
cudaFree(d_predicate_false_scan);
cudaFree(d_block_sum);
cudaFree(d_predicate_true_total_sum);
cudaFree(d_predicate_false_total_sum);
}

