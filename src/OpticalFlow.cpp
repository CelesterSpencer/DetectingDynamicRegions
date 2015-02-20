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
    std::cout << "Opt flow : \t" << timeSec << " sec" << std::endl;
}

void OpticalFlow::simplify(cv::gpu::GpuMat &in__opticalFlowMagnitude, cv::gpu::GpuMat &in__opticalFlowAngle, int numberOfMagnitudes, int numberOfAngles, cv::gpu::GpuMat &out__simplifiedFlowMagnitude, cv::gpu::GpuMat &out__simplifiedFlowAngle) {

    // get max magnitude
    double *minVal = new double;
    double *maxVal = new double;
    cv::gpu::minMax(in__opticalFlowMagnitude, minVal, maxVal);

    // simplify
    float maxMagnitude = (float)*maxVal;
    kernel.simplify(in__opticalFlowMagnitude, in__opticalFlowAngle, maxMagnitude, numberOfMagnitudes, numberOfAngles, out__simplifiedFlowMagnitude, out__simplifiedFlowAngle);

}
