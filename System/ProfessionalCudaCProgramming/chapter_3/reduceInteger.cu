#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int recursiveReduce(int *data, const int size) {
  if (size == 1) {
    return data[0];
  }

  const int stride = size / 2;

  for (int i = 0; i < stride; i++) {
    data[i] += data[i + stride];
  }

  return recursiveReduce(data, stride);
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x;

  if (idx >= n) {
    return;
  }

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      idata[tid] += idata[tid + stride];
    }

    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata,
                                     unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x;

  if (idx >= n) {
    return;
  }

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = 2 * stride * tid;
    if (index < blockDim.x) {
      idata[index] += idata[index + stride];
    }

    __syncthreads();
  }
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x;

  if (idx >= n) {
    return;
  }

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }

    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}

__global__ void reduceUnrolling(int *g_idata, int *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x * 2;

  if (idx + blockDim.x < n) {
    g_idata[idx] += g_idata[idx + blockDim.x];
  }

  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }

    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }

}

int main(int argc, char *argv[]) {
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("%s starting reduction at ", argv[0]);
  printf("device %d: %s", dev, deviceProp.name);
  cudaSetDevice(dev);

  bool bResult = false;

  int size = 1 << 28;
  printf("    with array size %d  ", size);

  int blocksize = 512;

  if (argc > 1) {
    blocksize = atoi(argv[1]);
  }

  dim3 block(blocksize, 1);
  dim3 grid((size + block.x - 1) / block.x, 1);
  printf("grid %d block %d\n", grid.x, block.x);

  size_t bytes = size * sizeof(int);
  int *h_idata = (int *)malloc(bytes);
  int *h_odata = (int *)malloc(grid.x * sizeof(int));
  int *tmp = (int *)malloc(bytes);

  for (int i = 0; i < size; i++) {
    h_idata[i] = (int)(rand() & 0xFF);
  }

  memcpy(tmp, h_idata, bytes);

  double iStart, iElaps;
  int gpu_sum = 0;

  int *d_idata = NULL;
  int *d_otata = NULL;
  cudaMalloc((void **)&d_idata, bytes);
  cudaMalloc((void **)&d_otata, grid.x * sizeof(int));

  // cpu reduction
  iStart = seconds();
  int cpu_sum = recursiveReduce(tmp, size);
  iElaps = seconds() - iStart;
  printf("cpu reduce      elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

  // kernel 1: reduceNeighbored
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  reduceNeighbored<<<grid, block>>>(d_idata, d_otata, size);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_otata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) {
    gpu_sum += h_odata[i];
  }
  printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block "
         "%d>>>\n",
         iElaps, gpu_sum, grid.x, block.x);

  // kernel 2: reduceNeighbored with less divergence
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  reduceNeighboredLess<<<grid, block>>>(d_idata, d_otata, size);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_otata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) {
    gpu_sum += h_odata[i];
  }
  printf("gpu Neighbored2 elapsed %f sec gpu_sum: %d <<<grid %d block "
         "%d>>>\n",
         iElaps, gpu_sum, grid.x, block.x);

  // kernel 3: reduceInterleaved
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  reduceInterleaved<<<grid, block>>>(d_idata, d_otata, size);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_otata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) {
    gpu_sum += h_odata[i];
  }
  printf("gpu Interleaved elapsed %f sec gpu_sum: %d <<<grid %d block "
         "%d>>>\n",
         iElaps, gpu_sum, grid.x, block.x);

  // kernel 4: unroll
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  reduceUnrolling<<<grid, block>>>(d_idata, d_otata, size);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_otata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++) {
    gpu_sum += h_odata[i];
  }
  printf("gpu unrolling   elapsed %f sec gpu_sum: %d <<<grid %d block "
         "%d>>>\n",
         iElaps, gpu_sum, grid.x, block.x);


  return 0;
}
