#ifndef _MAIN_H_
#define _MAIN_H_

#include "OpticalFlow.h"
#include "DepthMap.h"
#include "FlowVector3D.h"
#include "Segmentation.h"

#include <iostream>
#include <cstdio>
#include <time.h>
#include <Eigen/Dense>
#include "opencv2/opencv.hpp"

// OpenCV
#include "opencv2/opencv.hpp"
#include <opencv2/gpu/gpu.hpp>
// CUDA runtime.
#include <cuda.h>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

class Main {
private:


    void FlowToRGB(const cv::Mat &xFlow,cv::Mat &yFlow);
    void testOpticalFlow(cv::Mat &currentImg);
    int OpenCVCudaTest0();
    int main(int, char**);
};

#endif
