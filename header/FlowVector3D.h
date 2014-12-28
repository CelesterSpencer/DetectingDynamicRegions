#ifndef _FLOW_VECTOR_3D_H_
#define _FLOW_VECTOR_3D_H_

#include <iostream>

#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

/*!
 * \class A 3D flow vector is a finer assumption of the real world motion and is invariant against depth
 * \author Adrian Derstroff
 * \date 09.12.2014
 */
class FlowVector3D{
public:
    /*!
     * \brief Scales the optical flow vectors by the depth described in the depthmap
     * \param in__flowX dense optical flow field in x direction
     * \param in__flowY dense optical flow field in y direction
     * \param in__depthMap fine depth map that is used to scale the vectors
     * \param out__flow3DX resulting scale invariant vector field in x direction
     * \param out__flow3DY resulting scale invariant vector field in y direction
     */
    void calculate(cv::gpu::GpuMat &in__flowX, cv::gpu::GpuMat &in__flowY, cv::gpu::GpuMat &in__depthMap, cv::gpu::GpuMat &d__flow3DAngle, cv::gpu::GpuMat &d__flow3DMag);
private:

};

#endif
