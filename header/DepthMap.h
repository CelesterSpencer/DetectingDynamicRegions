#ifndef _DEPTH_MAP_H_
#define _DEPTH_MAP_H_

#include <opencv2/core/core.hpp>

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
     * \param in__opticalFlow dense optical flow field that is used to get a finer depth map
     * \param out__depthMap resulting fine depth map
     */
    void calculate(cv::Mat &in__inputFrame, cv::Mat &in__opticalFlow, cv::Mat &out__depthMap);
private:
    cv::Mat m__currentFrame;
    cv::Mat m__previousFrame;
    cv::Mat m__opticalFlow;
    cv::Mat m__motionHistoryImage;

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
