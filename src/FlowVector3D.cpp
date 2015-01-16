#include "FlowVector3D.h"
#include "scaleByDepth.h"

//----------------------------------------------------------------------------------------
// PUBLIC METHODS
//----------------------------------------------------------------------------------------

void FlowVector3D::calculate(cv::gpu::GpuMat &d__flowX, cv::gpu::GpuMat &d__flowY, cv::gpu::GpuMat &d__depthMap, cv::gpu::GpuMat &d__flow3DAngle, cv::gpu::GpuMat &d__flow3DMag) {
    const int64 start = cv::getTickCount();

    cv::gpu::GpuMat d__magnitude;
    std::cout << "created mat" << d__flowX.size() << d__flowY.size() << std::endl;
    cv::gpu::cartToPolar(d__flowX, d__flowY, d__magnitude, d__flow3DAngle, true);
    std::cout << "finished cart to polar" << std::endl;

    cv::gpu::multiply(d__magnitude, d__depthMap, d__flow3DMag);

    d__magnitude.release();
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Flow Vector 3D : \t" << timeSec << " sec" << std::endl;
}

