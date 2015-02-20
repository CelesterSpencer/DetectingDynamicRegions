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
    void calcMean(
            cv::gpu::GpuMat &in__isClassK,
            cv::gpu::GpuMat &in__YChannel, cv::gpu::GpuMat &in__UChannel, cv::gpu::GpuMat &in__VChannel,
            cv::gpu::GpuMat &in__uFLow, cv::gpu::GpuMat &in__vFlow,
            int in__classK,
            int in__numberOfPoints,
            float *outptr__meanVector);
    void calcCovarianzMatrix(
            cv::gpu::GpuMat &in__classes,
            cv::gpu::GpuMat &in__YChannel, cv::gpu::GpuMat &in__UChannel, cv::gpu::GpuMat &in__VChannel,
            cv::gpu::GpuMat &in__uFLow, cv::gpu::GpuMat &in__vFlow,
            float *inptr__meanVector,
            int in__classK,
            int in__numberOfPoints,
            cv::gpu::GpuMat &outptr__covarianzMatrix);
    void calculateFlowAndColorLikelihood(
            cv::gpu::GpuMat &in__YChannel, cv::gpu::GpuMat &in__UChannel, cv::gpu::GpuMat &in__VChannel,
            cv::gpu::GpuMat &in__uFlow, cv::gpu::GpuMat &in__vFlow,
            cv::gpu::GpuMat &in__covarianceMatrix,
            float *inptr__means,
            cv::gpu::GpuMat &out__flowLogLikelihoods, cv::gpu::GpuMat &out__colorLogLikelihoods,
            cv::gpu::GpuMat &out__maxFlowLikelihoods);
    void makeBinaryImage(cv::gpu::GpuMat &in__classes, int in__classK, cv::gpu::GpuMat &out__binaryImage);
    void matAdd(cv::gpu::GpuMat &gaussianImage, cv::gpu::GpuMat &out__spatialDeviations);
    void calculateLikelihood(
            cv::gpu::GpuMat &in__colorLogLikelihoods, cv::gpu::GpuMat &in__flowLogLikelihoods,
            cv::gpu::GpuMat &in__sumOfSpatialMeans, cv::gpu::GpuMat &in__maxFlowLogLikelihoods,
            int numberOfDataPoints, float sigma, int halfSearchRegion,
            cv::gpu::GpuMat &outptr__likelihoods);
    void getBiggestLikelihood(
            cv::gpu::GpuMat &ex__maxLikelihood, cv::gpu::GpuMat &ex__maxclass,
            cv::gpu::GpuMat &in__likelihood, int classK);
private:

};
