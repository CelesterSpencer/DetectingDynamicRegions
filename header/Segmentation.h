#ifndef _SEGMENTATION_H_
#define _SEGMENTATION_H_

#include <iostream>

#include <opencv2/core/core.hpp>
#include "opencv2/gpu/gpu.hpp"

#include "SegmentationKernel.h"
#include "Region.h"


/*!
 * \class Segments the input frame in several dynamic regions
 * \author Adrian Derstroff
 * \date 09.12.2014
 */
class Segmentation {
public:

    Segmentation() {}

    void calculate(cv::gpu::GpuMat &in__currentFrame, cv::gpu::GpuMat in__subtractedMagnitude, cv::gpu::GpuMat in__subtractedAngle, cv::Mat out__maskedRegions, std::vector<Region> &outptr__regions);
    void segment(cv::gpu::GpuMat &in__frameRGB, cv::gpu::GpuMat &in__flowX, cv::gpu::GpuMat &in__flowY, cv::gpu::GpuMat &out__segments);

private:
    SegmentationKernel kernel;

    bool m__isFirstSegmentation = true;


    cv::gpu::GpuMat m__classes;
    std::vector<int> m__numberOfPointsPerClass;
    int *m__classesX;
    int *m__classesY;
    int m__kMax = 1;
};

#endif
