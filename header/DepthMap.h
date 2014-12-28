#ifndef _DEPTH_MAP_H_
#define _DEPTH_MAP_H_

#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

/*!
 * \class Calculates a depth map in a coarse step and then refines it with the optical flow field
 * \author Adrian Derstroff
 * \date 09.12.2014
 */
class DepthMap {
public:
    /*!
     * \brief Calculate depth for every pixel depending on the optical flow
     * \param in__inputImage input image that is used to estimate a coarse depth map
     * \param in__opticalFlowX dense optical flow field in x direction
     * \param in__opticalFlowY dense optical flow field in y direction
     * \param out__depthMap resulting fine depth map
     */
    void calculate(cv::gpu::GpuMat &in__inputFrame, cv::gpu::GpuMat &in__opticalFlowX, cv::gpu::GpuMat &in__opticalFlowY, cv::gpu::GpuMat &out__depthMap);
private:
    cv::gpu::GpuMat m__currentFrame;
    cv::gpu::GpuMat m__previousFrame;
    cv::gpu::GpuMat m__opticalFlow;
    cv::gpu::GpuMat m__motionHistoryImage;

    void calcCoarseDepthMap();
    void calcClarity();
    void calcContrast();
    void calcColor();
    void calcFineDepthMap();
    void calcMotionHistoryImage();
    void segmentObject();
    void segmentBackground();
};

#endif
