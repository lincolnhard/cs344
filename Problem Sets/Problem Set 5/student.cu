/* Udacity HW5
Histogramming for Speed

The goal of this assignment is compute a histogram
as fast as possible.  We have simplified the problem as much as
possible to allow you to focus solely on the histogramming algorithm.

The input values that you need to histogram are already the exact
bins that need to be updated.  This is unlike in HW3 where you needed
to compute the range of the data and then do:
bin = (val - valMin) / valRange to determine the bin.

Here the bin is just:
bin = val

so the serial histogram calculation looks like:
for (i = 0; i < numElems; ++i)
histo[val[i]]++;

That's it!  Your job is to make it run as fast as possible!

The values are normally distributed - you may take
advantage of this fact in your implementation.

*/


#include "utils.h"
#include "reference.cpp"
#define NUM_VALS_PER_THREAD 21

__global__
void distribute_atomics_on_shmem_first(const unsigned int* const vals, //INPUT
unsigned int* const histo,      //OUPUT
const unsigned int numElems)
{
extern __shared__ unsigned int s_histo[];

s_histo[threadIdx.x] = 0;
__syncthreads();

for (int i = 0; i < NUM_VALS_PER_THREAD; i++) {
int id = blockDim.x * (i + NUM_VALS_PER_THREAD * blockIdx.x) + threadIdx.x;
if (id < numElems) {
unsigned int bin = vals[id];
atomicAdd(&s_histo[bin], 1);
}
}

__syncthreads();

// putting an if here makes it 20us slower
atomicAdd(&histo[threadIdx.x], s_histo[threadIdx.x]);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
unsigned int* const d_histo,      //OUTPUT
const unsigned int numBins,
const unsigned int numElems)
{
int numThreads2 = numBins;
int numBlocks2 = 1 + numElems / (NUM_VALS_PER_THREAD*numThreads2);
distribute_atomics_on_shmem_first<<<numBlocks2, numThreads2, sizeof(unsigned int)*numThreads2>>>(d_vals, d_histo, numElems);
cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

