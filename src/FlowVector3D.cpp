#include "FlowVector3D.h"

//----------------------------------------------------------------------------------------
// PUBLIC METHODS
//----------------------------------------------------------------------------------------

void FlowVector3D::calculate(cv::gpu::GpuMat &d__flowX, cv::gpu::GpuMat &d__flowY, cv::gpu::GpuMat &d__depthMap, cv::gpu::GpuMat &d__flow3DAngle, cv::gpu::GpuMat &d__flow3DMag) {
    const int64 start = cv::getTickCount();

    cv::gpu::GpuMat d__magnitude;
    cv::gpu::cartToPolar(d__flowX, d__flowY, d__magnitude, d__flow3DAngle, true);

    cv::gpu::multiply(d__magnitude, d__depthMap, d__flow3DMag);

    d__magnitude.release();
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Flow Vector 3D : \t" << timeSec << " sec" << std::endl;
}

