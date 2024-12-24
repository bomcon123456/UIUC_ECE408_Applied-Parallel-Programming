// LAB 2 FA24

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


// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  //@@ Implement matrix multiplication kernel here
  float Pvalue;
  int Row = blockDim.y * blockIdx.y + threadIdx.y;
  int Col = blockDim.x * blockIdx.x + threadIdx.x;
  
  if ((Row < numCRows) && (Col < numCColumns)) {
    Pvalue = 0;
    for (int i = 0; i < numAColumns; i++) {
      Pvalue += A[Row * numAColumns + i] * B[numBColumns * i + Col];
    }
    C[Row * numCColumns + Col] = Pvalue;
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
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

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
  int TileWidth = 4;
  dim3 dimGrid(ceil((1.0 * numCColumns) / TileWidth), ceil((1.0 * numCRows) / TileWidth), 1);
  dim3 dimBlock(TileWidth, TileWidth, 1);

  //@@ Launch the GPU Kernel here
  matrixMultiply<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows,
                                        numAColumns, numBRows,
                                        numBColumns, numCRows,
                                        numCColumns);
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
  //@@Free the hostC matrix
  delete[] hostC;
  return 0;
}

