// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan_stage3(float *input, float *output, int len) {
  int tx = threadIdx.x;
  int index = 2 * blockIdx.x * BLOCK_SIZE;
  if (blockIdx.x != 0 && index + tx < len){
    output[index + tx] += input[blockIdx.x - 1];
  }
  if (blockIdx.x != 0 && index + tx + BLOCK_SIZE < len){
    output[index + BLOCK_SIZE + tx] += input[blockIdx.x - 1];
  }
}

__global__ void scan_stage1(float *input, float *output, int len, float* stage2_sum) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2 * BLOCK_SIZE];
  int tx = threadIdx.x;
  int index = 2 * blockIdx.x * BLOCK_SIZE;
  int stride;

  if (index + tx < len) {
    T[tx] = input[index + tx];
  } else {
    T[tx] = 0.0f;
  }

  if (index + tx + BLOCK_SIZE < len) {
    T[tx + BLOCK_SIZE] = input[index + tx + BLOCK_SIZE];
  } else {
    T[tx + BLOCK_SIZE] = 0.0f;
  }

  stride = 1;
  while (stride <= BLOCK_SIZE) {
    __syncthreads();
    unsigned int index = (tx + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE) {
      T[index] += T[index - stride];
    }
    stride *= 2;
  } 
  
  stride = BLOCK_SIZE / 2;
  while (stride > 0) {
    __syncthreads();
    unsigned int index = (tx +1 ) * stride * 2 - 1;
    if ((index + stride) < 2 * BLOCK_SIZE) {
      T[index + stride] += T[index];
    }
    stride /= 2;
  }

  __syncthreads();
  if (index + tx < len) {
    output[index + tx] = T[tx];
  }
  if (index + tx + BLOCK_SIZE < len) {
    output[index + tx + BLOCK_SIZE] = T[tx + BLOCK_SIZE];
  }
  __syncthreads();
  
  if (tx == BLOCK_SIZE - 1 && stage2_sum != NULL) {
    stage2_sum[blockIdx.x] = T[2 * BLOCK_SIZE - 1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *stage2_sum;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&stage2_sum, ceil((1.0 * numElements) / (BLOCK_SIZE * 2)) * sizeof(float)));


  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 gridDim(ceil((1.0 * numElements) / (BLOCK_SIZE * 2)), 1, 1);
  dim3 gridDim_2(1, 1, 1);
  dim3 blockDim(BLOCK_SIZE, 1, 1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan_stage1<<<gridDim, blockDim>>>(deviceInput, deviceOutput, numElements, stage2_sum);
  cudaDeviceSynchronize();
  scan_stage1<<<gridDim_2, blockDim>>>(stage2_sum, stage2_sum, ceil((1.0 * numElements) / (BLOCK_SIZE * 2)), NULL);
  cudaDeviceSynchronize();
  scan_stage3<<<gridDim, blockDim>>>(stage2_sum, deviceOutput, numElements);
  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));


  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(stage2_sum);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

