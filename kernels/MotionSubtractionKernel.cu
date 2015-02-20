#include "MotionSubtractionKernel.h"

//---------------------------------------------------------------------------------------------------------------------------------
// GLOBAL MOTION SUBTRACTION
//---------------------------------------------------------------------------------------------------------------------------------

__global__ void dv__subtractMotion(float *flowMatX, float *flowMatY, int rows, int cols, size_t step, float *globalMotionX, float *globalMotionY, float *flowXSubtracted, float *flowYSubtracted) {

    // values
    int size = rows * cols;

    // get position within opticalflowfield
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y = idx / cols;
    int x = idx % cols;

    // bounds checking
    if (idx < size) {
        float xElement = ((float*)((char*)flowMatX + y*step))[x];
        float yElement = ((float*)((char*)flowMatY + y*step))[x];
        float xGlobalMotion = ((float*)((char*)globalMotionX + y*step))[x];
        float yGlobalMotion = ((float*)((char*)globalMotionY + y*step))[x];

        ((float*)((char*)flowXSubtracted + y*step))[x] = xElement - xGlobalMotion;
        ((float*)((char*)flowYSubtracted + y*step))[x] = yElement - yGlobalMotion;
    }

}

void MotionSubtractionKernel::subtractMotion(cv::gpu::GpuMat &flow3DX, cv::gpu::GpuMat &flow3DY, cv::gpu::GpuMat globalMotionX, cv::gpu::GpuMat globalMotionY, cv::gpu::GpuMat flowXSubtracted, cv::gpu::GpuMat flowYSubtracted) {

    // iterate over frame and subtract x and y
    int size = flow3DX.rows * flow3DX.cols;
    int threadSize = 1024;
    int blockSize = (size / threadSize)+1;

    dv__subtractMotion<<<blockSize, threadSize>>>(flow3DX.ptr<float>(), flow3DY.ptr<float>(), flow3DX.rows, flow3DX.cols, flow3DX.step, globalMotionX.ptr<float>(), globalMotionY.ptr<float>(), flowXSubtracted.ptr<float>(), flowYSubtracted.ptr<float>());

    cudaDeviceSynchronize();

}
