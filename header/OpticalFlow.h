#ifndef _OPTICAL_FLOW_H_
#define _OPTICAL_FLOW_H_

#include <opencv2/core/core.hpp>

/*!
 * \class Computes a dense optical field using the Brox et al. Optical flow implementation in OpenCV
 * \author Adrian Derstroff
 * \date 09.12.2014
 */
class OpticalFlow{
public:
    /*!
     * \brief computing a dense optical flow field using the input frame and the previous frame
     * \param in__inputFrame input image that is used to estimate the optical flow
     * \param out__opticalFlow returning dense optical flow field
     */
    void calculate(cv::Mat &in__inputFrame, cv::Mat &out__opticalFlow);
private:
    cv::Mat m__inputFrame;
    cv::Mat m__previousFrame;
};

#endif
