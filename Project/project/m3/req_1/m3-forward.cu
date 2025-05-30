#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

#define T 64  // T means the height of multiplication tile, aka the threads nums
#define U 16 // U means the width of multiplication tile, aka the number of elements processed by one thread
#define S (T / U) // S means the ratio between T and U
#define BLOCK_SIZE 256

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int W_grid = ceil((1.0*Width_out)/TILE_WIDTH);
    (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    // TODO: Insert your input matrix unrolling kernel code here
    #undef in_4d
}

__global__ void matrixMultiplyJointRegister(const float *A, const float *B, float *C,
                                            int numARows, int numAColumns,
                                            int numBRows, int numBColumns,
                                            int numCRows, int numCColumns)
{
    __shared__ float tileB[S][U];

    unsigned int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y;
    unsigned int row = by * blockDim.y + ty, col = bx * U;
    // Privatization of output variables
    float c_reg[U];
    // Initialize output values
    for (unsigned int outIdx = 0; outIdx < U; ++outIdx) {
        c_reg[outIdx] = 0;
    }

    // Iterate over tiles of B
    for (unsigned int tileId = 0; tileId < ceil(1.0 * numBRows / S); tileId++) {
        // Load tile of B into shared memory
        unsigned int i = threadIdx.y / U;
        unsigned int j = threadIdx.y % U;
        if (col + j < numBColumns && tileId * S + i < numBRows) {
            tileB[i][j] = B[(tileId * S + i) * numBColumns + col + j];
        } else {
            tileB[i][j] = 0.0f;
        }
        __syncthreads();  // Synchronize threads to ensure shared memory is loaded
        // Load A to register
        float a_reg;
        for (unsigned int idx = 0; idx < S; ++idx) {
            // Load tile of A matrix into register
            if (row < numARows && tileId * S + idx < numAColumns) {
                a_reg = A[row * numAColumns + tileId * S + idx];
            } else {
                a_reg = 0.0f;
            }
            // Loop over and update the output elements assigned to the thread
            for (unsigned int outIdx = 0; outIdx < U; ++outIdx) {
                c_reg[outIdx] += a_reg * tileB[idx][outIdx];
            }
        }
        __syncthreads();
    }

    for (unsigned int outIdx = 0; outIdx < U; ++outIdx) {
        if (row < numCRows && col + outIdx < numCColumns) {
            C[row * numCColumns + col + outIdx] = c_reg[outIdx];
        }
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    cudaMalloc( (void **) device_output_ptr, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float));
    cudaMalloc( (void **) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc( (void **) device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;

    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    dim3 gridDim_unroll(ceil(1.0 * Width_out / TILE_WIDTH) * ceil(1.0 * Height_out / TILE_WIDTH), Batch, 1);
    dim3 blockDim_unroll(TILE_WIDTH, TILE_WIDTH, 1);
    matrix_unrolling_kernel<<<gridDim_unroll, blockDim_unroll>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);
    cudaDeviceSynchronize();

    // TODO: Set the kernel dimensions and call the matmul kernel
    int Mask_col = Height_unrolled;
    int Mask_row = Map_out; 
    int Input_col = Width_unrolled;
    int Input_row = Height_unrolled;
    dim3 gridDim_matmul(ceil(1.0 * Input_col / U), ceil(1.0 * Mask_row / T),  1);
    dim3 blockDim_matmul(1, T, 1);
    matrixMultiplyJointRegister<<<gridDim_matmul, blockDim_matmul>>>(device_mask, unrolled_matrix, matmul_output, 
                                                                    Mask_row, Mask_col,
                                                                    Input_row, Input_col, Mask_row, Input_col);
    cudaDeviceSynchronize();
    // Permute the result of matrix multiplication
    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    cudaMemcpy(host_output, device_output, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    // TODO: Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
