#include "opencv2/gpu/gpu.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>

class OpticalFlowKernel{
public:
    void simplify(
            cv::gpu::GpuMat &in__opticalFlowMagnitude, cv::gpu::GpuMat &in__opticalFlowAngle,
            float maxMagnitude,
            int numberOfMagnitudes, int numberOfAngles,
            cv::gpu::GpuMat &out__simplifiedFlowMagnitude, cv::gpu::GpuMat &out__simplifiedFlowAngle);
private:

};
