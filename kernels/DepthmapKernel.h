#include "cuda.h"
#include "cuda_runtime.h"
#include "opencv2/gpu/gpu.hpp"
#include <iostream>

class DepthmapKernel {
public:
    void calculatedMeanAndClarityMap(cv::gpu::GpuMat &in__currentFrame, cv::gpu::GpuMat &out__meanMap, cv::gpu::GpuMat &out__clarityMap);
    void calculateContrastMap(cv::gpu::GpuMat &in__currentFrame, cv::gpu::GpuMat &in__meanMap, cv::gpu::GpuMat &out__contrastMap);
    void calculateClarityAndContrastPerImageblock(cv::gpu::GpuMat &in__clarityMap, cv::gpu::GpuMat &in__contrastMap, int in__numberOfImageblocks, int in__imageBlockSize, float *outptr__clarities, float *outptr__contrasts);
    void getMaxClarityAndContrast(float *inptr__clarities, float *inptr__contrasts, int in__numberOfImageblocks, float &maxClarity, float &maxContrast);
    void calculateDepthPerImageblock(float *inptr__clarities, float *inptr__contrasts, int size, float *outptr__depths);
    void calculateMeanCbAndCrValues(cv::gpu::GpuMat &in__cbMap, cv::gpu::GpuMat &in__crMap, int in__numberOfImageblocks, int in__imageBlockSize, float *outptr__meanCb, float *outptr__meanCr);
    void mergeNeighborImageblocks(float* inptr__meanCb, float* inptr__meanCr, int in__numberOfImageblocks, int imageBlockCols, float threshold, float *outptr__neighbors);
    void adjustDepthAndDistancelevel(float* inptr__neighbors, float *inptr__depths, int numberOfImageBlocks, float *outptr__adjustedDepths, int *outptr__distancelevels);
};
