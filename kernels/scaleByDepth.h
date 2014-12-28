#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "cuda.h"
#include "math.h"

void scaleDepth(cv::gpu::GpuMat flowX, cv::gpu::GpuMat flowY, cv::gpu::GpuMat depth, cv::gpu::GpuMat vec3D, int threadsize);
