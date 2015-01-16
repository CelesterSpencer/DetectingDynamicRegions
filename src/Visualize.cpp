#include "Visualize.h"

void Visualize::showFlow(cv::gpu::GpuMat &xFlow,cv::gpu::GpuMat &yFlow) {

    const int64 start = cv::getTickCount();

    //calculate angle and magnitude
    cv::gpu::GpuMat magnitude, angle;
    cv::gpu::cartToPolar(xFlow, yFlow, magnitude, angle, true);

    //build hsv image
    cv::gpu::GpuMat _hsv[3], hsv;
    cv::Mat onesMat = cv::Mat::ones(xFlow.size(), CV_32FC1);
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

    imshow("optical flow", out);
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Show image : \t" << timeSec << " sec" << std::endl;
}

void Visualize::show3DFlow(cv::gpu::GpuMat &flowAngle,cv::gpu::GpuMat &flowMagnitude) {

    const int64 start = cv::getTickCount();

    //build hsv image
    cv::gpu::GpuMat _hsv[3], hsv;
    cv::Mat onesMat = cv::Mat::ones(flowAngle.size(), CV_32FC1);
    cv::gpu::GpuMat onesGpu(onesMat);
    _hsv[0] = flowAngle;
    _hsv[1] = flowMagnitude;
    _hsv[2] = onesGpu;
    cv::gpu::merge(_hsv, 3, hsv);

    //convert to BGR
    cv::gpu::GpuMat bgr;
    cv::gpu::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    //transfer from device to host
    cv::Mat out;
    bgr.download(out);

    // display frame
    imshow("3d flow", out);
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Show 3Dflow : \t" << timeSec << " sec" << std::endl;
}

void Visualize::show3DFlow(cv::gpu::GpuMat &flowAngle,cv::gpu::GpuMat &flowMagnitude, std::string windowname) {

    const int64 start = cv::getTickCount();

    cv::gpu::GpuMat _hsv[3], hsv;
    cv::Mat onesMat = cv::Mat::ones(flowAngle.size(), CV_32F);
    cv::gpu::GpuMat onesGpu(onesMat);
    _hsv[0] = flowAngle;
    _hsv[1] = flowMagnitude;
    _hsv[2] = onesGpu;
    cv::gpu::merge(_hsv, 3, hsv);

    //convert to BGR
    cv::gpu::GpuMat bgr;
    cv::gpu::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    //transfer from device to host
    cv::Mat out;
    bgr.download(out);

    // display frame
    imshow(windowname, out);
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Show 3Dflow : \t" << timeSec << " sec" << std::endl;
}

void Visualize::showRegions(cv::Mat regions) {

    const int64 start = cv::getTickCount();

    // get 3 layers for BGR image
    cv::Mat r(regions.size(), CV_32FC1, cv::Scalar(0.0));
    cv::Mat g(regions.size(), CV_32FC1, cv::Scalar(0.0));
    cv::Mat bgrs[3];
    bgrs[0] = regions;
    bgrs[1] = g;
    bgrs[2] = r;

    // merge 3 layers to 1
    cv::Mat bgrMerged;
    cv::merge(bgrs,3, bgrMerged);

    // display frame
    imshow("regions", bgrMerged);
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Show regions : \t" << timeSec << " sec" << std::endl;
}
