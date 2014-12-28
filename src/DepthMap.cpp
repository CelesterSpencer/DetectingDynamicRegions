#include "DepthMap.h"

//----------------------------------------------------------------------------------------
// PUBLIC METHODS
//----------------------------------------------------------------------------------------

void DepthMap::calculate(cv::gpu::GpuMat &in__inputFrame, cv::gpu::GpuMat &in__opticalFlowX, cv::gpu::GpuMat &in__opticalFlowY, cv::gpu::GpuMat &out__depthMap) {
    //cv::Mat ones = cv::Mat(in__inputFrame.size(), CV_32FC1)*0.1;
    out__depthMap = cv::gpu::GpuMat(in__inputFrame.size(), CV_32FC1);//.upload(ones);
    out__depthMap.setTo(cv::Scalar::all(1));
}



//----------------------------------------------------------------------------------------
// PRIVATE METHODS
//----------------------------------------------------------------------------------------

void calcCoarseDepthMap() {

} 

void calcClarity() {

} 

void calcContrast() {

}

void calcColor() {

} 

void calcFineDepthMap() {

} 

void calcMotionHistoryImage() {

}

void segmentObject() {

}

void segmentBackground() {

}
