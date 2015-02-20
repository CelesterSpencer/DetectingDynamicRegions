#ifndef _VISUALIZE_H_
#define _VISUALIZE_H_

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "Region.h"

class Visualize {
public:

    void showFlow(cv::gpu::GpuMat &flowMagnitude, cv::gpu::GpuMat &flowAngle, std::string windowname);
    void showDepth(float *inptr__depth, int cols, int rows, int numberOfImageBlocks, int blockSize);
    void showMask(cv::Mat regions, std::string windowName);
    void showRegions(std::vector<Region> regions, int cols, int rows, std::string windowName);
private:

};

#endif
