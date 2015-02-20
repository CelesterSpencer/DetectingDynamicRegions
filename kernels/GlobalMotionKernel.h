#include "cuda.h"
#include "cuda_runtime.h"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <limits>

class GlobalMotionKernel {
public:
    void createCoarse3DFlow(cv::gpu::GpuMat &din__3DFlowX, cv::gpu::GpuMat &din__3DFlowY, int in__coarseLevel, cv::gpu::GpuMat &dout__coarse3DFlowX, cv::gpu::GpuMat &dout__coarse3DFlowY);
    void calculateSSDs(cv::gpu::GpuMat &din__3DFlowX, cv::gpu::GpuMat &din__3DFlowY, int startX, int startY, int endX, int endY, int in__w, float in__threshold, cv::gpu::GpuMat &dout__SSDs);
    void getPositionOfMinSSD(cv::gpu::GpuMat din__resultSSDs, int &outptr__xMinSSD, int &outptr__yMinSSD);
    void createSyntheticFlowField(cv::gpu::GpuMat &ex__syntheticFlowFieldX, cv::gpu::GpuMat &ex__syntheticFlowFieldY, int foeX, int foeY, float maxRotation, float translationX, float translationY);
    void calculateDivergenceOfFlowFields(std::vector<cv::gpu::GpuMat> &in__syntheticFlowFields, cv::gpu::GpuMat &in__realFlowFieldX, cv::gpu::GpuMat &in__realFlowFieldY, int &out__idxOfBestMatch);
};
