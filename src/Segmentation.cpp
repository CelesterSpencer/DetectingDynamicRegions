#include "Segmentation.h"

//----------------------------------------------------------------------------------------
// PUBLIC METHODS
//----------------------------------------------------------------------------------------

void Segmentation::calculate(cv::gpu::GpuMat &in__currentFrame, cv::gpu::GpuMat &in__flowVector3DAngle, cv::gpu::GpuMat &in__flowVector3DMagnitude, std::vector<Region> &out__dynamicRegions) {
    const int64 start = cv::getTickCount();

    calcGlobalMotion(in__flowVector3DAngle, in__flowVector3DMagnitude);

    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Segmentation : \t" << timeSec << " sec" << std::endl;
}



//----------------------------------------------------------------------------------------
// PRIVATE METHODS
//----------------------------------------------------------------------------------------

void Segmentation::calcGlobalMotion(cv::gpu::GpuMat &in__flowVector3DAngle, cv::gpu::GpuMat &in__flowVector3DMagnitude) {
    const int64 start = cv::getTickCount();

    SegmentationKernel kernel;
    kernel.getGlobalMotionHost(in__flowVector3DAngle, in__flowVector3DMagnitude, m__numberOfBins, 1024);

    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Global motion : \t" << timeSec << " sec" << std::endl;
}

void segmentDynamicObjects() {

}
