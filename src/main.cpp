// Prokject
#include "OpticalFlow.h"
#include "DepthMap.h"
#include "FlowVector3D.h"
#include "Segmentation.h"
#include "Visualize.h"
// C
#include <iostream>
#include <fstream>
// Opencv
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
// Cuda
#include <cuda.h>
#include <cuda_runtime.h>
// Eigen
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace cv::gpu;

String getStringFromInt(int val) {
    if (val < 10) {
        return "0" + to_string(val);
    }else {
        return to_string(val);
    }

}

String rootStr = "./res/frame";
String endStr = ".png";
int number = 0;
int nextFrame(cv::gpu::GpuMat &d__currentFrame) {
    // Get path name
    number = (number + 1) % 15;
    string path =  rootStr + getStringFromInt(number) + endStr;

    // Load image from path
    Mat frame = imread(path, IMREAD_GRAYSCALE);
    if (frame.empty()) {
        cerr << "Can't open image [" << path << "]" << endl;
        return -1;
    }

    // Convert mat
    GpuMat d_frame(frame);
    d_frame.convertTo(d__currentFrame, CV_32F, 1.0 / 255.0);
    return 0;
}

int main() {

    cv::gpu::DeviceInfo info = getDevice();
    int num_devices = cv::gpu::getCudaEnabledDeviceCount();
    std::cout << "GPU infos" << std::endl;
    std::cout << "Device count: " << num_devices << std::endl;
    std::cout << "Device info: " << info.name() << " version " <<  info.majorVersion() << "." << info.minorVersion() << std::endl;

    // Project modules
    OpticalFlow opticalFlowModule;
    DepthMap depthMapModule;
    FlowVector3D flowVector3DModule;
    Segmentation segmentationModule;
    Visualize visualizeModule("Dynamic regions");

    // Frames
    cv::gpu::GpuMat d__previousFrame;
    cv::gpu::GpuMat d__currentFrame;

    // Set previous frame
    nextFrame(d__previousFrame);

    // Optical flow
    cv::gpu::GpuMat d__flowX(d__previousFrame.size(), CV_32FC1);
    cv::gpu::GpuMat d__flowY(d__previousFrame.size(), CV_32FC1);

    while(1) {
        // Get next frame
        if (nextFrame(d__currentFrame) == -1) break;

        // Project pipeline
        opticalFlowModule.calculate(d__currentFrame, d__previousFrame, d__flowX, d__flowY);


        // Show result
        visualizeModule.showFlow(d__flowX, d__flowY);

        // saves  current frame in previous frame
        d__currentFrame.copyTo(d__previousFrame);
        if(waitKey(30) >= 0) break;
    }
    return 0;
}
