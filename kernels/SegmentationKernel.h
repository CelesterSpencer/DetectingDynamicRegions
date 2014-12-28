#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include "opencv2/gpu/gpu.hpp"
#include "cuda.h"
#include "cuda_runtime.h"

class SegmentationKernel{
public:
    void getGlobalMotionHost(cv::gpu::GpuMat &flow3DAngle, cv::gpu::GpuMat &flow3DMagnitude, int numberOfBins, int threadsize);
};
