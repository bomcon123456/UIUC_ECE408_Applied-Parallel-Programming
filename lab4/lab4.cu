#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define KERNEL_SIZE 3
#define TILE_WIDTH 4
//@@ Define constant memory for device kernel here
__constant__ float constKernel[KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int x_output = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int y_output = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int z_output = blockIdx.z * TILE_WIDTH + threadIdx.z;
  int radius = KERNEL_SIZE / 2;
  int x_input = x_output - radius;
  int y_input = y_output - radius;
  int z_input = z_output - radius;

  __shared__ float inputTile[TILE_WIDTH + KERNEL_SIZE - 1][TILE_WIDTH + KERNEL_SIZE - 1][TILE_WIDTH + KERNEL_SIZE - 1];

  if ((x_input >= 0) && (x_input < x_size) &&
      (y_input >= 0) && (y_input < y_size) &&
      (z_input >= 0) && (z_input < z_size)) {
    inputTile[tz][ty][tx] = input[z_input * x_size * y_size + y_input * x_size + x_input];
  } else {
    inputTile[tz][ty][tx] = 0.0f;
  }
  __syncthreads();
  float Pvalue = 0.0f;
  if (tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH) {
    for (int i = 0; i < KERNEL_SIZE; i++) {
      for (int j = 0; j < KERNEL_SIZE; j++) {
        for (int k = 0; k < KERNEL_SIZE; k++) {
          Pvalue += inputTile[i + tz][j + ty][k + tx] * constKernel[i * KERNEL_SIZE * KERNEL_SIZE + j * KERNEL_SIZE + k];
        }
      }
    }
    if (x_output >=0 && x_output < x_size && 
        y_output >= 0 && y_output < y_size &&
        z_output >= 0 && z_output < z_size) {
      output[z_output * y_size * x_size + y_output * x_size + x_output] = Pvalue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput, *deviceOutput;
  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc( (void **) &deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc( (void **) &deviceOutput, (inputLength - 3) * sizeof(float));


  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3], (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(constKernel, hostKernel, kernelLength * sizeof(float));

  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil((x_size * 1.0) / TILE_WIDTH), 
               ceil((y_size * 1.0) / TILE_WIDTH), 
               ceil((z_size * 1.0) / TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH + KERNEL_SIZE - 1, 
                TILE_WIDTH + KERNEL_SIZE - 1, 
                TILE_WIDTH + KERNEL_SIZE - 1);
  // dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();

  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);
  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

