// Project
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



std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

std::string getStringFromInt(int val) {
    if (val < 10) {
        return "0" + std::to_string(val);
    }else {
        return std::to_string(val);
    }

}

std::string rootStr = "./res/dot/frame";
std::string endStr = ".png";
int number = 0;
int nextFrame(cv::gpu::GpuMat &d__currentFrame) {
    const int64 start = cv::getTickCount();
    // Get path name
    number = (number + 1) % 8;
    std::string path =  rootStr + getStringFromInt(number) + endStr;
    //std::cout << "open " << path << std::endl;

    // Load image from path
    cv::Mat frame = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (frame.empty()) {
        std::cerr << "Can't open image [" << path << "]" << std::endl;
        return -1;
    }

    cv::Size size(frame.cols/2, frame.rows/2);
    resize(frame, frame, size , 0, 0, cv::INTER_CUBIC);

    // Convert mat
    cv::gpu::GpuMat d__frame(frame);
    d__frame.convertTo(d__currentFrame, CV_32FC1, 1.0 / 255.0);
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Load image : \t" << timeSec << " sec" << std::endl;
    return 0;
}

int main() {

    cv::gpu::DeviceInfo info = cv::gpu::getDevice();
    int num_devices = cv::gpu::getCudaEnabledDeviceCount();
    std::cout << "GPU infos" << std::endl;
    std::cout << "Device count: " << num_devices << std::endl;
    std::cout << "Device info: " << info.name() << " version " <<  info.majorVersion() << "." << info.minorVersion() << std::endl;

    // Project modules
    OpticalFlow opticalFlowModule;
    DepthMap depthMapModule;
    FlowVector3D flowVector3DModule;
    Segmentation segmentationModule(16);
    Visualize visualizeModule("Dynamic regions");

    // Frames
    cv::gpu::GpuMat d__previousFrame;
    cv::gpu::GpuMat d__currentFrame;

    // Set previous frame
    nextFrame(d__previousFrame);

    // Optical flow
    cv::gpu::GpuMat d__flowX(d__previousFrame.size(), CV_32FC1);
    cv::gpu::GpuMat d__flowY(d__previousFrame.size(), CV_32FC1);
    // DepthMap
    cv::gpu::GpuMat d__depthMap(d__previousFrame.size(), CV_32FC1);
    // 3DFlow
    cv::gpu::GpuMat d__flowAngle(d__previousFrame.size(), CV_32FC1);
    cv::gpu::GpuMat d__flowMagnitude(d__previousFrame.size(), CV_32FC1);
    // Segmentation
    std::vector<Region> regions;

    while(1) {
        const int64 start = cv::getTickCount();

        // Get next frame
        if (nextFrame(d__currentFrame) == -1) break;

        // Project pipeline
        opticalFlowModule.calculate(d__currentFrame, d__previousFrame, d__flowX, d__flowY);
        depthMapModule.calculate(d__currentFrame, d__flowX, d__flowY, d__depthMap);
        flowVector3DModule.calculate(d__flowX, d__flowY, d__depthMap, d__flowAngle, d__flowMagnitude);
        std::cout << type2str(d__flowAngle.type()) << std::endl;
        segmentationModule.calculate(d__currentFrame, d__flowAngle, d__flowMagnitude, regions);

        // Show result
        //visualizeModule.showFlow(d__flowX, d__flowY);
        visualizeModule.show3DFlow(d__flowAngle, d__flowMagnitude);

        // saves  current frame in previous frame
        d__currentFrame.copyTo(d__previousFrame);

        // total time
        const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
        std::cout << "Total time : \t" << timeSec << " sec" << std::endl;
        if(cv::waitKey(30) >= 0) break;
        std::cout << "----------------------------------------------------" << std::endl;
    }
    return 0;
}
