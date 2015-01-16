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

    Segmentation(int numberOfAngles, int numberOfMagnitudes) {
        m__numberOfAngles = numberOfAngles;
        m__numberOfMagnitudes = numberOfMagnitudes;
        m__numberOfBins = numberOfAngles * numberOfMagnitudes;
    }

    /*!
     * \brief Subtract global motion from the 3D flow vector field and then group all pixels with similar 3D flow vector to a dynamic region
     * \param in__flowVector3D 3D flow vector field that is used  to estimate the global motion and finally helps to segment the frame into several regions
     * \param out__dynamicRegions returning all segmented regions
     */
    void calculate(cv::gpu::GpuMat &in__currentFrame, cv::gpu::GpuMat &in__flowVector3DAngle, cv::gpu::GpuMat &in__flowVector3DMagnitude, cv::gpu::GpuMat &out__subtractedX, cv::gpu::GpuMat &out__subtractedY, cv::Mat &out__regions, std::vector<Region> &out__dynamicRegions);

private:
    cv::Mat m__flowVector3D;
    std::vector<cv::Vec3d> m__flowFieldLibrary;

    void calcGlobalMotion(cv::gpu::GpuMat &in__flowVector3DAngle, cv::gpu::GpuMat &in__flowVector3DMagnitude, float &out__globalAngle, float &out__globalMagnitude);
    void subtractGlobalMotion(cv::gpu::GpuMat &in__flowVector3DX, cv::gpu::GpuMat &in__flowVector3DY, float in__globalX, float in__globalY, cv::gpu::GpuMat &out__subtractedX, cv::gpu::GpuMat &out__subtractedY);
    void segmentDynamicObjects(cv::gpu::GpuMat &flowAngleSubtracted, cv::gpu::GpuMat &flowMagnitudeSubtracted, cv::Mat &flowAngleSubtractedHost);
    int m__numberOfAngles;
    int m__numberOfMagnitudes;
    int m__numberOfBins;
    float m__lengthPerMagnitude;
    int m__threadSize = 1024;
};

#endif
