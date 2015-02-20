#include "OpticalFlowKernel.h"

//---------------------------------------------------------------------------------------------------------------------------------
// Simplify
//---------------------------------------------------------------------------------------------------------------------------------
__global__ void dv__simplify(
        float *inptr__opticalFlowMagnitude, float *inptr__opticalFlowAngle,
        size_t step,
        int cols, int rows,
        float maxMagnitude,
        int numberOfMagnitudes, int numberOfAngles,
        float *outptr__simplifiedFlowMagnitude, float *outptr__simplifiedFlowAngle) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int size = rows * cols;
    int x = idx % cols;
    int y = idx / cols;
    float degreePerAngle = 360.0f / numberOfAngles;
    float magnitude = maxMagnitude / numberOfMagnitudes;



    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= size) return;



    //---------------------------------------------------------------------------------------------
    // simplify
    //---------------------------------------------------------------------------------------------
    float oldMagnitude = ((float*)((char*)inptr__opticalFlowMagnitude + y*step))[x];
    float oldAngle = ((float*)((char*)inptr__opticalFlowAngle + y*step))[x];
    ((float*)((char*)outptr__simplifiedFlowMagnitude + y*step))[x] = ((int)(oldMagnitude / magnitude)) * magnitude;
    ((float*)((char*)outptr__simplifiedFlowAngle + y*step))[x] = (((int)(oldAngle / degreePerAngle)) * degreePerAngle);

}

void OpticalFlowKernel::simplify(
        cv::gpu::GpuMat &in__opticalFlowMagnitude, cv::gpu::GpuMat &in__opticalFlowAngle,
        float maxMagnitude,
        int numberOfMagnitudes, int numberOfAngles,
        cv::gpu::GpuMat &out__simplifiedFlowMagnitude, cv::gpu::GpuMat &out__simplifiedFlowAngle) {

    int size = in__opticalFlowMagnitude.cols * in__opticalFlowMagnitude.rows;
    int threadSize = 1024;
    int blockSize = (size / threadSize) + 1;

    dv__simplify<<<blockSize, threadSize>>>(
            in__opticalFlowMagnitude.ptr<float>(), in__opticalFlowAngle.ptr<float>(),
            in__opticalFlowMagnitude.step,
            in__opticalFlowMagnitude.cols, in__opticalFlowMagnitude.rows,
            maxMagnitude,
            numberOfMagnitudes, numberOfAngles,
            out__simplifiedFlowMagnitude.ptr<float>(), out__simplifiedFlowAngle.ptr<float>());
}
