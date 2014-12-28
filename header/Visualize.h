#ifndef _VISUALIZE_H_
#define _VISUALIZE_H_

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

class Visualize {
public:
    Visualize(std::string windowName) {m__windowName = windowName; }

    void showFlow(cv::gpu::GpuMat &xFlow,cv::gpu::GpuMat &yFlow);
    void show3DFlow(cv::gpu::GpuMat &flowAngle,cv::gpu::GpuMat &flowMagnitude);
private:
    std::string m__windowName;
};

#endif
