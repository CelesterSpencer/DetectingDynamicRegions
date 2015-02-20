#ifndef _DEPTH_MAP_H_
#define _DEPTH_MAP_H_

#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "DepthmapKernel.h"
#include "Visualize.h"

/*!
 * \class Calculates a depth map in a coarse step and then refines it with the optical flow field
 * \author Adrian Derstroff
 * \date 09.12.2014
 */
class DepthMap {
public:

    void setNumberOfImagesBeingProcessed(int numberOfImagesBeingProessed) {
        m_numberOfImagesBeingProcessed = numberOfImagesBeingProessed;
    }

    /*!
     * \brief Calculate depth for every pixel depending on the optical flow
     * \param in__inputImage input image that is used to estimate a coarse depth map
     * \param in__opticalFlowX dense optical flow field in x direction
     * \param in__opticalFlowY dense optical flow field in y direction
     * \param out__depthMap resulting fine depth map
     */
    void calculate(cv::gpu::GpuMat &in__inputFrameGray, cv::gpu::GpuMat &in__inputFrameRGB, cv::gpu::GpuMat &in__opticalFlowX, cv::gpu::GpuMat &in__opticalFlowY, int in__imageblockSize, cv::gpu::GpuMat &out__depthMap);

private:
    cv::gpu::GpuMat m__currentFrame;
    cv::gpu::GpuMat m__previousFrame;
    cv::gpu::GpuMat m__opticalFlow;
    cv::gpu::GpuMat m__motionHistoryImage;

    int m__numberOfImageblocks = -1;

    DepthmapKernel kernel;

    Visualize m__visualizeModule;

    int number = 1;

    int m_numberOfImagesBeingProcessed = -1;

    void calcCoarseDepthMap(cv::gpu::GpuMat &in__currentFrameGray, cv::gpu::GpuMat &in__currentFrameRGB, int in__imageblockSize, int numberOfImageblocks, float in__threshold, float *outptr__depth);
    void calcFineDepthMap();
    void drawGpuMat(cv::gpu::GpuMat d__mat, std::string name);
    void printArray(float *dptr__array, int size, std::string name);
};

#endif
