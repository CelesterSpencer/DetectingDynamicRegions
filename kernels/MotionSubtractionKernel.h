#include "opencv2/gpu/gpu.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>

class MotionSubtractionKernel{
public:
    void subtractMotion(cv::gpu::GpuMat &flow3DX, cv::gpu::GpuMat &flow3DY, cv::gpu::GpuMat globalMotionX, cv::gpu::GpuMat globalMotionY, cv::gpu::GpuMat flowXSubtracted, cv::gpu::GpuMat flowYSubtracted);

private:

};
