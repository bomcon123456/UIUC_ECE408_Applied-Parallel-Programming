#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
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
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int h_unroll;
    unsigned int w_unroll;
    int h = (bx / W_grid) * TILE_WIDTH + ty;
    int w = (bx % W_grid) * TILE_WIDTH + tx;
    int b = by;
    if(h>=0 && h<Height_out && w>=0 && w<Width_out){
        for (int c = 0;  c < Channel; c++){ 
            int w_base = c * K * K; 
            for (int p = 0; p < K; p++){
                for (int q = 0; q < K; q++){
                    h_unroll = w_base + p * K + q;
                    w_unroll = h * Width_out + w + b * Height_out * Width_out;
                    float input_value = 0.0f;
                    if((h+p) < Height && (w+q) < Width){
                        input_value = in_4d(b, c, h+p, w+q);
                        output[h_unroll * Batch * Height_out * Width_out + w_unroll] = input_value;
                    }
                     
                }  
            }     
        }
    }    
    #undef in_4d
}

// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = val;
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
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    float *unrolled_matrix, *matmul_output;

    cudaHostRegister((void *) host_input, sizeof(float) * Batch * Channel * Height * Width, cudaHostRegisterDefault);
    cudaHostRegister((void *) host_output, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1), cudaHostRegisterDefault);
    // cudaHostRegister((void *) host_mask, sizeof(float) * Map_out * Channel * K * K, cudaHostRegisterDefault);

    cudaMalloc((void **) device_output_ptr, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1));
    cudaMalloc((void **) device_input_ptr,  sizeof(float) * Batch * Channel * Height * Width);
    cudaMalloc((void **) device_mask_ptr,   sizeof(float) * Map_out * Channel * K * K);
    cudaMemcpy(*device_mask_ptr, host_mask, sizeof(float) * Map_out * Channel * K * K, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
    cudaMalloc((void **)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    int num_streams = 4;
    cudaStream_t stream_list[num_streams];

    if (Batch % num_streams != 0) {
        num_streams = 1;
    }

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&stream_list[i]);
    }

    for(int  i =0 ; i < num_streams; i++){
        size_t offset_input = (size_t) i * Batch * Channel * Height * Width;
        offset_input /= num_streams;
        cudaMemcpyAsync(*device_input_ptr + offset_input, host_input + offset_input, sizeof(float) * Batch * Channel * Height * Width / num_streams, cudaMemcpyHostToDevice, stream_list[i]);
    }

    for(int  i =0 ; i < num_streams; i++){
        size_t offset_input = (size_t) i * Batch * Channel * Height * Width;
        size_t offset_unrolled = (size_t) i * Batch * Channel * K * K * Height_out * Width_out;
        offset_input /= num_streams;
        offset_unrolled /= num_streams;
        dim3 block_unroll(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 grid_unroll(ceil((1.0*Height_out)/TILE_WIDTH) * ceil((1.0*Width_out)/TILE_WIDTH), Batch / num_streams, 1);
        matrix_unrolling_kernel<<<grid_unroll, block_unroll, 0, stream_list[i]>>>(*device_input_ptr + offset_input, unrolled_matrix + offset_unrolled, Batch / num_streams, Channel, Height, Width, K);
    }

    for(int  i =0 ; i < num_streams; i++){
        size_t offset_unrolled = (size_t) i * Batch * Channel * K * K * Height_out * Width_out;
        size_t offset_matmul = (size_t) i * Batch * Map_out * Height_out * Width_out;
        offset_unrolled /= num_streams;
        offset_matmul /= num_streams;
        int Height_unrolled = Channel * K * K;
        int Width_unrolled = Batch / num_streams * Height_out * Width_out;
        int A_col = Height_unrolled;
        int A_row = Map_out; 
        dim3 matrixMultiply_grid(ceil((1.0 * Width_unrolled) / TILE_WIDTH), ceil((1.0 * A_row) / TILE_WIDTH), 1);
        dim3 matrixMultiply_block(TILE_WIDTH, TILE_WIDTH, 1);
        matrixMultiplyShared<<<matrixMultiply_grid, matrixMultiply_block, 0, stream_list[i]>>>(*device_mask_ptr, unrolled_matrix + offset_unrolled, matmul_output + offset_matmul, A_row, A_col,
                                Height_unrolled, Width_unrolled, A_row, Width_unrolled);  
    }

    for(int  i =0 ; i < num_streams; i++){
        // Permute the result of matrix multiplication
        size_t offset_output = (size_t) i * Batch * Map_out * Height_out * Width_out;
        size_t offset_matmul = (size_t) i * Batch * Map_out * Height_out * Width_out;
        offset_output /= num_streams;
        offset_matmul /= num_streams;
        dim3 permute_grid_dim((Height_out * Width_out - 1) / BLOCK_SIZE + 1, Batch / num_streams, 1);
        matrix_permute_kernel<<<permute_grid_dim, BLOCK_SIZE, 0, stream_list[i]>>>(
            matmul_output + offset_matmul, *device_output_ptr + offset_output, Map_out, Batch / num_streams, Height_out * Width_out);
    }

    for(int  i =0 ; i < num_streams; i++){
        size_t offset_output = (size_t) i * Batch * Map_out * Height_out * Width_out;
        offset_output /= num_streams;
        // Copy the output chunk back to the host
        cudaMemcpyAsync((void*)(host_output + offset_output), *device_output_ptr + offset_output,
                        sizeof(float)* Batch * Map_out * Height_out * Width_out / num_streams, cudaMemcpyDeviceToHost, stream_list[i]);
    }

    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(stream_list[i]);
        cudaStreamDestroy(stream_list[i]);
    }

    // Free memory
    cudaFree(unrolled_matrix);
    cudaFree(matmul_output);
    cudaFree(*device_output_ptr);
    cudaFree(*device_input_ptr);
    cudaFree(*device_mask_ptr);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

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