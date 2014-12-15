#include "OpticalFlow.h"

//----------------------------------------------------------------------------------------
// PUBLIC METHODS
//----------------------------------------------------------------------------------------
OpticalFlow::OpticalFlow() {

}

void OpticalFlow::calculate(cv::gpu::GpuMat &in__currentFrame, cv::gpu::GpuMat &in__previousFrame, cv::gpu::GpuMat &out__opticalFlowX, cv::gpu::GpuMat &out__opticalFlowY) {
    const int64 start = cv::getTickCount();

    brox(in__previousFrame, in__currentFrame, out__opticalFlowX, out__opticalFlowY);

    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Brox : " << timeSec << " sec" << std::endl;
}
