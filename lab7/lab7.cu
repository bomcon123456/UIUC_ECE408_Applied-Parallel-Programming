// Histogram Equalization

#include <wb.h>

#define BLOCK_SIZE       16
#define HISTOGRAM_LENGTH 256

//@@ insert code here

__global__ void toChar(float *input, unsigned char *output, int len) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < len){
        output[index] = (unsigned char)(255 * input[index]);
    }  
}

__global__ void toGreyScale(unsigned char *input, unsigned char *output, int len) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned char r, g, b;
    if (index < len){
        r = input[3 * index];
        g = input[3 * index + 1];
        b = input[3 * index + 2];
        output[index] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

__global__ void toHistogram(unsigned char *input, unsigned int *output, int len) {
    __shared__ unsigned int histogram[HISTOGRAM_LENGTH];
    unsigned tx = threadIdx.x;
    unsigned int index = tx + blockIdx.x * blockDim.x;

    if (tx < HISTOGRAM_LENGTH){
        histogram[tx] = 0;
    }
    __syncthreads();
    if (index < len){
        atomicAdd(&(histogram[input[index]]), 1); // subtotal in each block
    }
    __syncthreads();
    if (tx < HISTOGRAM_LENGTH){
        atomicAdd(&(output[tx]), histogram[tx]); // add the subtotal to the global memory
    }
}

__global__ void toCDF(unsigned int *input, float *output, int len) {
    __shared__ float CDF[HISTOGRAM_LENGTH];
    unsigned tx = threadIdx.x;
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride;
    if (index < HISTOGRAM_LENGTH){
        CDF[index] = input[index];
    }
    __syncthreads();

    stride = 1;
    while (stride < HISTOGRAM_LENGTH){
        __syncthreads();
        int i = (tx + 1) * stride * 2 - 1;
        if (i < HISTOGRAM_LENGTH){
            CDF[i] += CDF[i - stride];
        }
        stride *= 2;
    }
    __syncthreads();

    stride = HISTOGRAM_LENGTH / 4;
    while (stride > 0){
        __syncthreads();
        int i = (tx + 1) * stride * 2 - 1;
        if (i + stride < HISTOGRAM_LENGTH){
            CDF[i + stride] += CDF[i];
        }
        stride /= 2;
    }
    __syncthreads();

    if (index < HISTOGRAM_LENGTH){
        output[index] = (float)CDF[index] * 1.0 / len;
    }  
}

__global__ void equalization(unsigned char *image, float *CDF, int len){
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    float x, correctColor;
    if (index < len){
        x = 255.0 * (CDF[image[index]] - CDF[0]) / (1 - CDF[0]);
        correctColor = min(max(x, 0.0f), 255.0f);
        image[index] = (unsigned char)correctColor;
    }
}

__global__ void toFloat(unsigned char *input, float *output, int len){
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < len){
        output[index] = (float)input[index] * 1.0 / 255;
    }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData, *CDF;
  unsigned char *deviceChar, *deviceGreyChar;
  unsigned int *histogram;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  //@@ insert code here
  cudaMalloc((void **)&deviceInputImageData, imageChannels * imageHeight * imageWidth * sizeof(float));
  cudaMalloc((void **)&CDF, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **)&deviceChar, imageChannels * imageHeight * imageWidth * sizeof(unsigned char));
  cudaMalloc((void **)&deviceGreyChar, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&histogram, HISTOGRAM_LENGTH * sizeof(unsigned int));

  cudaMemcpy(deviceInputImageData, hostInputImageData, imageChannels * imageHeight * imageWidth * sizeof(float), cudaMemcpyHostToDevice);

  dim3 gridDim_char((imageChannels * imageHeight * imageWidth + BLOCK_SIZE * BLOCK_SIZE - 1) / (BLOCK_SIZE * BLOCK_SIZE), 1, 1);
  dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE, 1, 1);
  toChar<<<gridDim_char, blockDim>>>(deviceInputImageData, deviceChar, imageChannels * imageHeight * imageWidth);  

  dim3 gridDim((imageWidth * imageHeight + BLOCK_SIZE * BLOCK_SIZE - 1) / (BLOCK_SIZE * BLOCK_SIZE), 1, 1);
  toGreyScale<<<gridDim, blockDim>>>(deviceChar, deviceGreyChar, imageHeight * imageWidth);
  cudaDeviceSynchronize();

  toHistogram<<<gridDim, blockDim>>>(deviceGreyChar, histogram, imageWidth * imageHeight);
  cudaDeviceSynchronize();

  toCDF<<<1, HISTOGRAM_LENGTH>>>(histogram, CDF, imageWidth * imageHeight);
  cudaDeviceSynchronize();

  equalization<<<gridDim_char, blockDim>>>(deviceChar, CDF, imageChannels * imageHeight * imageWidth);
  cudaDeviceSynchronize();

  toFloat<<<gridDim_char, blockDim>>>(deviceChar, deviceInputImageData, imageChannels * imageHeight * imageWidth);
  cudaDeviceSynchronize();
  //@@ insert code here
  cudaMemcpy(hostOutputImageData, deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceChar);
  cudaFree(deviceGreyChar);
  cudaFree(histogram);
  cudaFree(CDF);
  return 0;
}

