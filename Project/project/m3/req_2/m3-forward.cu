#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void fused_matrix_unrolling_kernel(const float *input, const float *mask, float *output, 
                                              int Batch, int Map_out, int Channel, 
                                              int Height, int Width, 
                                              int K,
                                              int unrolled_input_height, int unrolled_input_width,
                                              int unrolled_mask_height, int unrolled_mask_width,
                                              int unrolled_output_height, int unrolled_output_width) {
    __shared__ float subTile_input[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTile_mask[TILE_WIDTH][TILE_WIDTH];

    const int Width_out = (Width - K) + 1;

    int b = blockIdx.z; //The index of the image in the batch.
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx; // The row number of the current element in matrix C
                                                                // The column number of the current element in matrix C
    float output_val = 0.0;
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    //Loop on each pair of tiles from A and B, 
    //Note that here (unrolled_mask_width-1) / TILE_WIDTH + 1 == ceil(unrolled_mask_width/TILE_WIDTH)
    for (int i = 0; i < (unrolled_mask_width - 1) / TILE_WIDTH + 1; i++){  //unrolled_mask_width == unrolled_input_height
    //load subtiles of A and B from the global memory to the shared memory of this block
        unsigned int mask_column = i * TILE_WIDTH + tx;
        if (row >= unrolled_mask_height || mask_column >= unrolled_mask_width){ //Boundary condition            
            subTile_mask[ty][tx] = 0;
        } else {
            subTile_mask[ty][tx] = mask[row * unrolled_mask_width + mask_column];
        }
        if (i * TILE_WIDTH + ty >= unrolled_input_height || col >= unrolled_input_width) { //Boundary condition condition
            subTile_input[ty][tx] = 0;
        } else {
            int c = (i * TILE_WIDTH + ty) / (K * K);    //The channel index of the input feature map, not it is integer division
            int h_out = col / Width_out;
            int w_out = col % Width_out;
            int w_base = c * K * K;
            int p = (i * TILE_WIDTH + ty - w_base) / K;
            int q = (i * TILE_WIDTH + ty - w_base) % K;
            // Row index of the unrolled matrix for the thread to write
            // the input element into for the current iteration
            if ((h_out + p < Height) && (w_out + q < Width)){
                subTile_input[ty][tx] = in_4d(b, c, h_out + p, w_out + q);
            } else {
                subTile_input[ty][tx] = 0;
            }
        }
        __syncthreads();  //Synchronize the threads, we need to wait all threads finish loading the data
        //Use the subtiles to calculate the partial sum for the element, and accumulate into output_val
        if (row < unrolled_output_height && col < unrolled_output_width){ //Boundary condition
            for (int k=0; k<TILE_WIDTH; k++){
                output_val += subTile_mask[ty][k] * subTile_input[k][tx];
            }
        }
        __syncthreads();  //Synchronize the threads, we need to wait all threads finish computation, so that we can get the correct answer
    }

    if (row < unrolled_output_height && col < unrolled_output_width){ //Boundary condition
        output[b * (unrolled_output_height * unrolled_output_width) + row * unrolled_output_width + col] = output_val;  //Store the result element back
    }

    #undef out_4d
    #undef in_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    cudaMalloc( (void **) device_output_ptr, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float));
    cudaMalloc( (void **) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc( (void **) device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask,
                                             const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;    //Output height for one output feature map
    const int Width_out = Width - K + 1;    //Output width for one output feature map
    int unrolled_input_height = Channel * K * K;
    int unrolled_input_width = Height_out * Width_out;
    int unrolled_mask_height = Map_out;
    int unrolled_mask_width = Channel * K * K;
    // const int input_grid = ceil((1.0 * Height_out * Width_out) / TILE_WIDTH);
    // const int map_grid = ceil((1.0 * Map_out) / TILE_WIDTH);    //Number of tiles in height for all output feature maps in an image

    dim3 dimGrid(ceil((1.0 * Height_out * Width_out) / TILE_WIDTH), ceil((1.0 * Map_out) / TILE_WIDTH), Batch);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);   //The dimension of a block (a tile)

    //Launch the convolution forward kernel.
    fused_matrix_unrolling_kernel<<<dimGrid, dimBlock>>>(device_input, device_mask, device_output, Batch, Map_out, Channel, Height, Width, K,
                                                        unrolled_input_height, unrolled_input_width, unrolled_mask_height, unrolled_mask_width,
                                                        unrolled_mask_height, unrolled_input_width);
    cudaDeviceSynchronize();
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