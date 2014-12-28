#include "Visualize.h"

void Visualize::showFlow(cv::gpu::GpuMat &xFlow,cv::gpu::GpuMat &yFlow) {

    const int64 start = cv::getTickCount();

    //calculate angle and magnitude
    cv::gpu::GpuMat magnitude, angle;
    cv::gpu::cartToPolar(xFlow, yFlow, magnitude, angle, true);

//    cv::gpu::minMax(magnitude, minPtr, maxPtr);
//    cv::gpu::GpuMat  normalizedMag;
//    cv::gpu::normalize(magnitude,normalizedMag);

    //build hsv image
    cv::gpu::GpuMat _hsv[3], hsv;
    cv::Mat onesMat = cv::Mat::ones(xFlow.size(), CV_32F);
    cv::gpu::GpuMat onesGpu(onesMat);
    _hsv[0] = angle;
    _hsv[1] = magnitude;
    _hsv[2] = onesGpu;
    cv::gpu::merge(_hsv, 3, hsv);

    //convert to BGR
    cv::gpu::GpuMat bgr;
    cv::gpu::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    //transfer from device to host
    cv::Mat out;
    bgr.download(out);

    imshow(m__windowName, out);
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Show image : \t" << timeSec << " sec" << std::endl;
}

void Visualize::show3DFlow(cv::gpu::GpuMat &flowAngle,cv::gpu::GpuMat &flowMagnitude) {

    const int64 start = cv::getTickCount();


    //cv::gpu::minMax(magnitude, minPtr, maxPtr);
    cv::gpu::GpuMat  normalizedMag;
    cv::gpu::normalize(flowMagnitude,normalizedMag, 0, 255, cv::NORM_MINMAX);

    cv::gpu::GpuMat _hsv[3], hsv;
    cv::Mat onesMat = cv::Mat::ones(flowAngle.size(), CV_32F);
    cv::gpu::GpuMat onesGpu(onesMat);
    _hsv[0] = flowAngle;
    _hsv[1] = normalizedMag;
    _hsv[2] = onesGpu;
    cv::gpu::merge(_hsv, 3, hsv);

    //convert to BGR
    cv::gpu::GpuMat bgr;
    cv::gpu::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    //transfer from device to host
    cv::Mat out;
    bgr.download(out);

    imshow(m__windowName, out);
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Show image : \t" << timeSec << " sec" << std::endl;
}
