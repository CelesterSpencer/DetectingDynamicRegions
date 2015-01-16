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







// basketball, car, dot, kids, yosemit
std::string rootStr = "./res/kids/frame";
std::string endStr = ".png";
int number = 0;
int nextFrame(cv::gpu::GpuMat &d__currentFrameGray, cv::gpu::GpuMat &d__currentFrameRGB) {
    const int64 start = cv::getTickCount();




    //-----------------------------------------------------------------------------------------------------
    // Get path name
    //-----------------------------------------------------------------------------------------------------
    number = (number + 1) % 8;
    std::string path =  rootStr + getStringFromInt(number) + endStr;
    //std::cout << "open " << path << std::endl;




    //-----------------------------------------------------------------------------------------------------
    // Load image from path
    //-----------------------------------------------------------------------------------------------------
    cv::Mat colorFrame = cv::imread(path, cv::IMREAD_COLOR);
    if (colorFrame.empty()) {
        std::cerr << "Can't open image [" << path << "]" << std::endl;
        return -1;
    }



    //-----------------------------------------------------------------------------------------------------
    // Convert mat
    //-----------------------------------------------------------------------------------------------------
    cv::Mat grayscaleFrame;
    cv::cvtColor(colorFrame, grayscaleFrame, CV_RGB2GRAY);




    //-----------------------------------------------------------------------------------------------------
    //scale image to half size
    //-----------------------------------------------------------------------------------------------------
    cv::Size size(colorFrame.cols/2, colorFrame.rows/2);
    resize(colorFrame, colorFrame, size , 0, 0, cv::INTER_CUBIC);
    resize(grayscaleFrame, grayscaleFrame, size , 0, 0, cv::INTER_CUBIC);

    cv::gpu::GpuMat d__frameRGB(colorFrame);
    cv::gpu::GpuMat d__frameGray(grayscaleFrame);
    d__frameRGB.convertTo(d__currentFrameRGB, CV_32FC3, 1.0 / 255.0);
    d__frameGray.convertTo(d__currentFrameGray, CV_32FC1, 1.0 / 255.0);



    //-----------------------------------------------------------------------------------------------------
    // release memory
    //-----------------------------------------------------------------------------------------------------
    d__frameGray.release();
    d__frameRGB.release();



    //-----------------------------------------------------------------------------------------------------
    // execution time
    //-----------------------------------------------------------------------------------------------------
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Load image : \t" << timeSec << " sec" << std::endl;
    return 0;
}







#define NUMBEROFANGLES 8
#define NUMBEROFMAGNITUDES 10

int main() {

    try {
    //-----------------------------------------------------------------------------------------------------
    // print gpu specific information
    //-----------------------------------------------------------------------------------------------------
    cv::gpu::DeviceInfo info = cv::gpu::getDevice();
    int num_devices = cv::gpu::getCudaEnabledDeviceCount();
    std::cout << "GPU infos" << std::endl;
    std::cout << "Device count: " << num_devices << std::endl;
    std::cout << "Device info: " << info.name() << " version " <<  info.majorVersion() << "." << info.minorVersion() << std::endl;



    //-----------------------------------------------------------------------------------------------------
    // setup Project modules
    //-----------------------------------------------------------------------------------------------------
    OpticalFlow opticalFlowModule;
    DepthMap depthMapModule;
    FlowVector3D flowVector3DModule;
    Segmentation segmentationModule(NUMBEROFANGLES, NUMBEROFMAGNITUDES);
    Visualize visualizeModule("Dynamic regions");




    //-----------------------------------------------------------------------------------------------------
    // Frames
    //-----------------------------------------------------------------------------------------------------
    cv::gpu::GpuMat d__previousFrameGray;
    cv::gpu::GpuMat d__currentFrameGray;
    cv::gpu::GpuMat d__currentFrameRGB;




    //-----------------------------------------------------------------------------------------------------
    // Set previous frame
    //-----------------------------------------------------------------------------------------------------
    nextFrame(d__previousFrameGray, d__currentFrameRGB);

    while(1) {
        //-----------------------------------------------------------------------------------------------------
        // start timer
        //-----------------------------------------------------------------------------------------------------
        const int64 start = cv::getTickCount();




        //-----------------------------------------------------------------------------------------------------
        // Get next frame
        //-----------------------------------------------------------------------------------------------------
        if (nextFrame(d__currentFrameGray, d__currentFrameRGB) == -1) break;
        std::cout << "frame sizes: " << d__currentFrameGray.size() << ", " << d__currentFrameRGB.size() << ", " << d__previousFrameGray.size() << std::endl;
        std::cout << "rgb frame type is " << type2str(d__currentFrameRGB.type()) << std::endl;




        //-----------------------------------------------------------------------------------------------------
        // Project pipeline
        //-----------------------------------------------------------------------------------------------------

        // show current frame
        cv::Mat currentFrame, currentFrameRGB;
        d__currentFrameGray.download(currentFrame);
        d__currentFrameRGB.download(currentFrameRGB);
        imshow("current frame rgb", currentFrameRGB);

        // initialize Optical flow variables
        cv::gpu::GpuMat d__opticalFlowX(d__previousFrameGray.size(), CV_32FC1);
        cv::gpu::GpuMat d__opticalFlowY(d__previousFrameGray.size(), CV_32FC1);
        opticalFlowModule.calculate(d__currentFrameGray, d__previousFrameGray, d__opticalFlowX, d__opticalFlowY);
        visualizeModule.showFlow(d__opticalFlowX, d__opticalFlowY);

        // initialize DepthMap variables
        cv::gpu::GpuMat d__depthMap(d__previousFrameGray.size(), CV_32FC1);
        depthMapModule.calculate(d__currentFrameGray,d__currentFrameRGB, d__opticalFlowX, d__opticalFlowY, d__depthMap);
        std::cout << "finished depthmap" << std::endl;

        // initialize 3DFlow variables
//        cv::gpu::GpuMat d__flow3DAngle(d__previousFrameGray.size(), CV_32FC1);
//        cv::gpu::GpuMat d__flow3DMagnitude(d__previousFrameGray.size(), CV_32FC1);
//        std::cout << "start 3dflow" << std::endl;
//        flowVector3DModule.calculate(d__opticalFlowX, d__opticalFlowY, d__depthMap, d__flow3DAngle, d__flow3DMagnitude);
//        visualizeModule.show3DFlow(d__flow3DAngle, d__flow3DMagnitude, "3D flow");
//        std::cout << "finished 3dflow" << std::endl;
//        d__opticalFlowX.release();
//        d__opticalFlowY.release();
//        d__depthMap.release();

        // initialize Segmentation variables
//        std::vector<Region> segments;
//        cv::gpu::GpuMat d__subtracktedAngle(d__previousFrameGray.size(), CV_32FC1);
//        cv::gpu::GpuMat d__subtracktedMagnitude(d__previousFrameGray.size(), CV_32FC1);
//        cv::Mat regions(d__previousFrameGray.size(), CV_32FC1);
//        segmentationModule.calculate(d__currentFrameGray, d__flow3DAngle, d__flow3DMagnitude, d__subtracktedAngle, d__subtracktedMagnitude, regions, segments);
//        visualizeModule.show3DFlow(d__subtracktedAngle, d__subtracktedMagnitude, "flow without global motion");
//        visualizeModule.showRegions(regions);
//        d__flow3DAngle.release();
//        d__flow3DMagnitude.release();
//        d__subtracktedAngle.release();
//        d__subtracktedMagnitude.release();

        // show masked regions
//        regions.convertTo(regions, CV_8U);
//        cv::Mat maskedFrame;
//        currentFrameRGB.copyTo(maskedFrame, regions);
//        imshow("masked frame", maskedFrame);
//        regions.release();
//        maskedFrame.release();



        //-----------------------------------------------------------------------------------------------------
        // saves  current frame in previous frame
        //-----------------------------------------------------------------------------------------------------
//        d__currentFrameGray.copyTo(d__previousFrameGray);



        //-----------------------------------------------------------------------------------------------------
        // total time
        //-----------------------------------------------------------------------------------------------------.
        const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
        std::cout << "Total time : \t" << timeSec << " sec" << std::endl;
        if(cv::waitKey(30) >= 0) break;
        std::cout << "----------------------------------------------------" << std::endl;

    }
    }catch(cv::Exception & e) {
        std::cout << "Caught cv::Error: " << std::endl;
        std::cout << e.what() << std::endl;
        std::cout << e.msg << std::endl;
        std::cout << e.code << std::endl;
    }

    return 0;
}
