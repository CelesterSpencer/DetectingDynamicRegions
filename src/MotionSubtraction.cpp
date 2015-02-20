#include "MotionSubtraction.h"

void MotionSubtraction::subtractGlobalMotion(cv::gpu::GpuMat &in__flowVector3DAngle, cv::gpu::GpuMat &in__flowVector3DMagnitude, cv::gpu::GpuMat &in__globalMotionX, cv::gpu::GpuMat &in__globalMotionY, cv::gpu::GpuMat &out__subtractedAngle, cv::gpu::GpuMat &out__subtractedMagnitude) {
    const int64 start = cv::getTickCount();

    //----------------------------------------------------------------------------------------
    // convert from magnitde/angle to x/y
    //----------------------------------------------------------------------------------------
    cv::gpu::GpuMat subtractedX(in__flowVector3DAngle.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::gpu::GpuMat subtractedY(in__flowVector3DAngle.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::gpu::GpuMat flowVector3DX, flowVector3DY;
    cv::gpu::polarToCart(in__flowVector3DMagnitude, in__flowVector3DAngle, flowVector3DX, flowVector3DY, true);



    //---------------------------------------------------------------------------------------------------------------------------------
    // segmentation
    //---------------------------------------------------------------------------------------------------------------------------------
    kernel.subtractMotion(flowVector3DX, flowVector3DY, in__globalMotionX, in__globalMotionY, subtractedX, subtractedY);



    //----------------------------------------------------------------------------------------
    // convert from x/y to magnitde/angle
    //----------------------------------------------------------------------------------------
    cv::gpu::cartToPolar(subtractedX, subtractedY, out__subtractedMagnitude, out__subtractedAngle, true);



    //---------------------------------------------------------------------------------------------------------------------------------
    // display computation time
    //---------------------------------------------------------------------------------------------------------------------------------
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Motion subtr : \t" << timeSec << " sec" << std::endl;

}
