#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include "opencv2/gpu/gpu.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>

class SegmentationKernel{
public:
    SegmentationKernel() {m__threadSize = 1024; }
    void setThreadSize(int threadSize) {m__threadSize = threadSize; }


    void fillBins(float* inptr__flowVector3DMagnitude, float* inptr__flowVector3DAngle, size_t in__flowVector3DMagnitudeStep, size_t in__flowVector3DAngleStep,
                  int in__cols, int in__rows, int in__numberOfMagnitudes, int in__numberOfAngles, float in__lengthPerMagnitude, int* outptr__bins);
    void sumUpBins(int in__tempSize, bool in__isOdd, int in__numberOfBins, int* outptr__bins);
    void globalMotionSubtractionHost(cv::gpu::GpuMat &flow3DX, cv::gpu::GpuMat &flow3DY, float globalX, float globalY, cv::gpu::GpuMat flowXSubtracted, cv::gpu::GpuMat flowYSubtracted);


private:
    int m__threadSize;
};
