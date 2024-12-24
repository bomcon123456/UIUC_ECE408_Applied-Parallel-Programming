#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int column = bx * TILE_WIDTH + tx;
  float Cvalue = 0;

  for (int t = 0; t < ceil((numAColumns * 1.0) / TILE_WIDTH); t++) {
    if (row < numARows && (t * TILE_WIDTH + tx) < numAColumns){
      subTileA[ty][tx] = A[row * numAColumns + t * TILE_WIDTH + tx];
    } else {
      subTileA[ty][tx] = 0.0;
    }

    if ((ty + t * TILE_WIDTH) < numBRows && column < numBColumns) {
      subTileB[ty][tx] = B[(ty + t * TILE_WIDTH) * numBColumns + column];
    } else {
      subTileB[ty][tx] = 0.0;
    }
    __syncthreads();
    for (int i = 0; i < TILE_WIDTH; i++) {
      Cvalue += subTileA[ty][i] * subTileB[i][tx];
    }
    __syncthreads();
  }
  if (row < numCRows && column < numCColumns) {
    C[row * numCColumns + column] = Cvalue;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  hostC = new float[numCRows * numCColumns];

  //@@ Allocate GPU memory here
  float *deviceA, *deviceB, *deviceC;
  int inputSizeA = numARows * numAColumns * sizeof(float);
  int inputSizeB = numBRows * numBColumns * sizeof(float);
  int outputSizeC = numCRows * numCColumns * sizeof(float);
  cudaMalloc( (void **) &deviceA, inputSizeA);
  cudaMalloc( (void **) &deviceB, inputSizeB);
  cudaMalloc( (void **) &deviceC, outputSizeC);

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, inputSizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, inputSizeB, cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil((1.0 * numCColumns) / TILE_WIDTH), ceil((1.0 * numCRows) / TILE_WIDTH), 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC,
                                              numARows, numAColumns,
                                              numBRows, numBColumns,
                                              numCRows, numCColumns);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, outputSizeC, cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@ Free the hostC matrix
  delete[] hostC;
  return 0;
}
