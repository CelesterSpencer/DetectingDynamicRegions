#ifndef _OPTICAL_FLOW_H_
#define _OPTICAL_FLOW_H_

#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "OpticalFlowKernel.h"

/*!
 * \class Computes a dense optical field using the Brox et al. Optical flow implementation in OpenCV
 * \author Adrian Derstroff
 * \date 09.12.2014
 */
class OpticalFlow{
private:
    float m_alpha = 0.197f;
    float m_gamma = 50.0f;
    float m_scaleFactor = 0.8f;
    int m_innerIterations = 20;
    int m_outerIterations = 77;
    int m_solverIterations = 10;

    OpticalFlowKernel kernel;

    // (alpha, gamma, pyramid scale factor, number of inner iterations, number of outer iterations, number of basic solver iterations)
    cv::gpu::BroxOpticalFlow brox = cv::gpu::BroxOpticalFlow(m_alpha, m_gamma, m_scaleFactor, m_innerIterations, m_outerIterations, m_solverIterations);
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

    void setAlpha(float alpha) {
        m_alpha = alpha;
    }

    void setGamma(float gamma) {
        m_gamma = gamma;
    }

    void setScaleFactor(float scaleFactor) {
        m_scaleFactor = scaleFactor;
    }

    void setInnerIterations(int innerIterations) {
        m_innerIterations = innerIterations;
    }

    void setOuterIterations(int outerIterations) {
        m_outerIterations = outerIterations;
    }

    void setSolverIterations(int solverIterations) {
        m_solverIterations = solverIterations;
    }

    void update() {
        brox = cv::gpu::BroxOpticalFlow(m_alpha, m_gamma, m_scaleFactor, m_innerIterations, m_outerIterations, m_solverIterations);
    }

    void simplify(cv::gpu::GpuMat &out__opticalFlowMagnitude, cv::gpu::GpuMat &out__opticalFlowAngle, int numberOfMagnitudes, int numberOfAngles, cv::gpu::GpuMat &out__simplifiedFlowMagnitude, cv::gpu::GpuMat &out__simplifiedFlowAngle);
};

#endif
