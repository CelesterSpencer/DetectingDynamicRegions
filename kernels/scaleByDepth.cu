//#include "scaleByDepth.h"
#include "cuda.h"
#include "opencv2/gpu/gpu.hpp"

extern "C" {
    void scaleDepth();
}

__global__ void scaleByDepth(float *flowX, float *flowY, float *depth, int size) {

    // get position within opticalflowfield
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // get value for calculated position
    if (i >= size) return; // check bounds

    // scalar product
    float vectorlength = sqrt(flowX[i] * flowX[i] + flowY[i] * flowY[i]) ;

    printf("ich bin thread %d \n", i);
}

void scaleDepth(cv::gpu::GpuMat flowX, cv::gpu::GpuMat flowY, cv::gpu::GpuMat depth, cv::gpu::GpuMat vec3D, int threadsize) {
    int size = flowX.rows * flowX.cols;
    int blocksize = size / threadsize;
    float *dataX = (float*)flowX.data;
    float *dataY = (float*)flowY.data;
    float *dataD = (float*)depth.data;

    float *d__dataX;
    float *d__dataY;
    float *d__dataD;
    cudaMalloc((float**)&d__dataX, sizeof(float) * size);
    cudaMalloc((float**)&d__dataY, sizeof(float) * size);
    cudaMalloc((float**)&d__dataD, sizeof(float) * size);
    cudaMemcpy(d__dataX, dataX, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d__dataY, dataY, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d__dataD, dataD, sizeof(float) * size, cudaMemcpyHostToDevice);

    scaleByDepth<<<blocksize, threadsize>>>(d__dataX, d__dataY, d__dataD, size);
    cudaDeviceSynchronize();

    cudaMemcpy(vec3D.data, d__dataD, sizeof(float) * size, cudaMemcpyDeviceToHost);

    cudaFree(d__dataX);
    cudaFree(d__dataY);
    cudaFree(d__dataD);
}
