#ifndef _MOTION_SUBTRACTION_H_
#define _MOTION_SUBTRACTION_H_

#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "MotionSubtractionKernel.h"


/*!
 * \class Calculates a depth map in a coarse step and then refines it with the optical flow field
 * \author Adrian Derstroff
 * \date 09.12.2014
 */
class MotionSubtraction {
public:
    void subtractGlobalMotion(cv::gpu::GpuMat &in__flowVector3DAngle, cv::gpu::GpuMat &in__flowVector3DMagnitude, cv::gpu::GpuMat &in__globalMotionX, cv::gpu::GpuMat &in__globalMotionY, cv::gpu::GpuMat &out__subtractedAngle, cv::gpu::GpuMat &out__subtractedMagnitude);

private:
    MotionSubtractionKernel kernel;
};

#endif
