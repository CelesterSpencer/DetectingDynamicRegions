#ifndef _FLOW_VECTOR_3D_H_
#define _FLOW_VECTOR_3D_H_

#include <opencv2/core/core.hpp>

/*!
 * \class A 3D flow vector is a finer assumption of the real world motion and is invariant against depth
 * \author Adrian Derstroff
 * \date 09.12.2014
 */
class FlowVector3D{
public:
    /*!
     * \brief Scales the optical flow vectors by the depth described in the depthmap
     * \param in__inputFrame input image that is used to estimate a coarse depth map
     * \param in__opticalFlow dense optical flow field
     * \param in__depthMap fine depth map that is used to scale the vectors
     * \param out__flowVector3D resulting vector field with vectors scaled by their depth
     */
    void calculate(cv::Mat &in__inputFrame, cv::Mat &in__opticalFlow, cv::Mat &in__depthMap, cv::Mat &out__flowVector3D);
private:
    cv::Mat m__inputFrame;
    cv::Mat m__opticalFlow;
    cv::Mat m__depthMap;
};

#endif
