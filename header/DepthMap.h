#ifndef _DEPTH_MAP_H_
#define _DEPTH_MAP_H_

#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "DepthmapKernel.h"

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
    void DepthMap::calculate(cv::gpu::GpuMat &in__inputFrameGray, cv::gpu::GpuMat &in__inputFrameRGB, cv::gpu::GpuMat &in__opticalFlowX, cv::gpu::GpuMat &in__opticalFlowY, cv::gpu::GpuMat &out__depthMap);
private:
    cv::gpu::GpuMat m__currentFrame;
    cv::gpu::GpuMat m__previousFrame;
    cv::gpu::GpuMat m__opticalFlow;
    cv::gpu::GpuMat m__motionHistoryImage;

    DepthmapKernel kernel;

    void calcCoarseDepthMap(cv::gpu::GpuMat &in__currentFrameGray, cv::gpu::GpuMat &in__currentFrameRGB, float in__threshold);
    void calcFineDepthMap();
    void printFreeSpace() {
        size_t free;
        size_t total;
        cudaMemGetInfo(&free, &total);
        double freeMb = ((double)free)/1024.0/1024.0;
        double totalMb = ((double)total)/1024.0/1024.0;
        std::cout << freeMb << "MB free, " << totalMb << "MB total" << std::endl;
    }
};

#endif
