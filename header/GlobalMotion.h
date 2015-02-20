#ifndef _GLOBAL_MOTION_H_
#define _GLOBAL_MOTION_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include "GlobalMotionKernel.h"

/*!
 * \class Estimates the translation and rotation of the camera
 * \author Adrian Derstroff
 * \date 30.01.2015
 */
class GlobalMotion {
public:

    void calculate(cv::gpu::GpuMat &in__3DFlowX, cv::gpu::GpuMat &in__3DFlowY, int w1, int w2, float threshold, int coarseLevel, cv::gpu::GpuMat &out__globalMotionX, cv::gpu::GpuMat &out__globalMotionY);

    void getGlobalMotionParameters(float &ex__translationX, float &ex__translationY, float &ex__angle) {
        ex__translationX = m_translationX;
        ex__translationY = m_translationY;
        ex__angle = m_angle;
    }

private:

    GlobalMotionKernel kernel;

    float m_translationX = -1;
    float m_translationY = -1;
    float m_angle = -1;

    void drawFOE(cv::gpu::GpuMat d__coarse3DFlowX, cv::gpu::GpuMat d__coarse3DFlowY, int centerX, int centerY, int startX, int startY, int endX, int endY );
    void drawFlow(cv::Mat coarseFlowMatMagnitude, cv::Mat coarseFlowMatAngle, std::string windowName);

};

#endif
