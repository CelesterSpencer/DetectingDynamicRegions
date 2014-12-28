#ifndef _OPTICAL_FLOW_H_
#define _OPTICAL_FLOW_H_

#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

/*!
 * \class Computes a dense optical field using the Brox et al. Optical flow implementation in OpenCV
 * \author Adrian Derstroff
 * \date 09.12.2014
 */
class OpticalFlow{
private:
    // (alpha, gamma, pyramid scale factor, number of inner iterations, number of outer iterations, number of basic solver iterations)
    cv::gpu::BroxOpticalFlow brox = cv::gpu::BroxOpticalFlow(0.197f, 50.0f, 0.8f, 20, 77, 10);
public:
    OpticalFlow();
    /*!
     * \brief computing a dense optical flow field using the input frame and the previous frame
     * \param in__currentFrame input current image
     * \param in__previousFrame input previous frame
     * \param out__opticalFlowX returning dense optical flow field in x direction
     * \param out__opticalFlowY returning dense optical flow field in y direction
     */
    void calculate(cv::gpu::GpuMat &in__currentFrame, cv::gpu::GpuMat &in__previousFrame, cv::gpu::GpuMat &out__opticalFlowX, cv::gpu::GpuMat &out__opticalFlowY);
};

#endif
