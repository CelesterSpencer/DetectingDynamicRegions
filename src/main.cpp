// Project
#include "OpticalFlow.h"
#include "DepthMap.h"
#include "FlowVector3D.h"
#include "GlobalMotion.h"
#include "MotionSubtraction.h"
#include "Segmentation.h"
#include "Visualize.h"
#include "Tester.h"
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

//#define _TESTING_ true


bool isStopedByUser = false;

//-----------------------------------------------------------------------------------------------------
// helpermethod to get information about the type of a Mat
//-----------------------------------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------------------------------
// helpermethod to get a string for the number of the current frame
//-----------------------------------------------------------------------------------------------------
std::string getStringFromInt(int val) {
    if (val < 10) {
        return "0" + std::to_string(val);
    }else {
        return std::to_string(val);
    }

}


//-----------------------------------------------------------------------------------------------------
// get next gray and RGB frame from image series
//-----------------------------------------------------------------------------------------------------
// basketball, car, dot, kids, yosemit
std::string rootStr = "./res/TestData2/Frame0";
std::string endStr = ".png";
int number = 0;
int nextFrame(int numberOfImagesBeigProcessed, cv::gpu::GpuMat &dout__currentFrameGray, cv::gpu::GpuMat &dout__currentFrameRGB) {
    const int64 start = cv::getTickCount();




    //-----------------------------------------------------------------------------------------------------
    // Get path name
    //-----------------------------------------------------------------------------------------------------
    number = (number + 1) % numberOfImagesBeigProcessed;
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
    d__frameRGB.convertTo(dout__currentFrameRGB, CV_32FC3, 1.0 / 255.0);
    d__frameGray.convertTo(dout__currentFrameGray, CV_32FC1, 1.0 / 255.0);



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

void getDepth(int in_imageNumber, int in_numberOfImages, cv::gpu::GpuMat groundTruthMap) {

    //-----------------------------------------------------------------------------------------------------
    // Get path name
    //-----------------------------------------------------------------------------------------------------
    int number = in_imageNumber % in_numberOfImages;
    std::string rootStr = "./res/TestData2/Depth0";
    std::string numberStr = "";
    std::string endStr = ".png";
    if (number < 10) {
        numberStr = "0" + std::to_string(number);
    }else {
        numberStr = std::to_string(number);
    }
    std::string path =  rootStr + numberStr + endStr;
    //std::cout << "open " << path << std::endl;

    //-----------------------------------------------------------------------------------------------------
    // Load image from path
    //-----------------------------------------------------------------------------------------------------
    cv::Mat depthFrame = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (depthFrame.empty()) {
        std::cerr << "Can't open image [" << path << "]" << std::endl;
    }

    //-----------------------------------------------------------------------------------------------------
    //scale image to half size
    //-----------------------------------------------------------------------------------------------------
    cv::Size size(depthFrame.cols/2, depthFrame.rows/2);
    resize(depthFrame, depthFrame, size , 0, 0, cv::INTER_CUBIC);

    cv::Mat normalizedDepth;
    depthFrame.convertTo(normalizedDepth, CV_32FC1, 1.0f / 255);

    groundTruthMap.upload(normalizedDepth);

}





#define NUMBEROFANGLES 8
#define NUMBEROFMAGNITUDES 10
#define IMAGEBLOCKSIZE 9

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
    GlobalMotion globalMotion;
    MotionSubtraction motionSubtraction;
    Segmentation segmentationModule;
    Visualize visualizeModule;
    #ifdef _TESTING_
        Tester tester;
    #endif



    //-----------------------------------------------------------------------------------------------------
    // Frames
    //-----------------------------------------------------------------------------------------------------
    cv::gpu::GpuMat d__previousFrameGray;
    cv::gpu::GpuMat d__currentFrameGray;
    cv::gpu::GpuMat d__currentFrameRGB;



    //-----------------------------------------------------------------------------------------------------
    // counter for number of images being processed before trying different parameters
    //-----------------------------------------------------------------------------------------------------
    int numberOfImagesBeingProcessed = 50;
    int counter = 1;
    depthMapModule.setNumberOfImagesBeingProcessed(numberOfImagesBeingProcessed);
    #ifdef _TESTING_
        tester.readGroundTruthData("./res/TestData/FramesInfo.txt", numberOfImagesBeingProcessed);
    #endif




    //-----------------------------------------------------------------------------------------------------
    // Set previous frame
    //-----------------------------------------------------------------------------------------------------
    nextFrame(numberOfImagesBeingProcessed, d__previousFrameGray, d__currentFrameRGB);



    while(!isStopedByUser) {
        //-----------------------------------------------------------------------------------------------------
        // Update parameters
        //-----------------------------------------------------------------------------------------------------
        //optical flow
            // alpha
            // gamma
            // scale factor
            // inner iterations
            // outer iterations
            // solver iterations
        //depth map
            // image block size
        //global motion
            // matched filter 1 radius
            // matched filter 2 radius
            // threshold
            // coarse level
        //segmentation
            // number of angles
            // number of magnitudes
        counter = 1;


        while(1) {
            //-----------------------------------------------------------------------------------------------------
            // start timer
            //-----------------------------------------------------------------------------------------------------
            const int64 start = cv::getTickCount();
            #ifdef _TESTING_
                float testSum = 0;
                float temp_sum = 0;
            #endif



            //-----------------------------------------------------------------------------------------------------
            // Get next frame
            //-----------------------------------------------------------------------------------------------------
            if (nextFrame(numberOfImagesBeingProcessed, d__currentFrameGray, d__currentFrameRGB) == -1) break;
            #ifdef _TESTING_
                tester.printFrameNumber(counter);
            #endif



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
            cv::gpu::GpuMat d__opticalFlowMagnitude, d__opticalFlowAngle;
            cv::gpu::cartToPolar(d__opticalFlowX, d__opticalFlowY, d__opticalFlowMagnitude, d__opticalFlowAngle, true);
            visualizeModule.showFlow(d__opticalFlowMagnitude, d__opticalFlowAngle, "Opticalflow");

            // simplified flow
            cv::gpu::GpuMat d__simplifiedFlowMagnitude(d__previousFrameGray.size(), CV_32FC1);
            cv::gpu::GpuMat d__simplifiedFlowAngle(d__previousFrameGray.size(), CV_32FC1);
            opticalFlowModule.simplify(d__opticalFlowMagnitude, d__opticalFlowAngle, 10, 16, d__simplifiedFlowMagnitude, d__simplifiedFlowAngle);
            visualizeModule.showFlow(d__simplifiedFlowMagnitude, d__simplifiedFlowAngle, "Simplified flow");
            cv::gpu::GpuMat d__simplifiedFlowX(d__previousFrameGray.size(), CV_32FC1);
            cv::gpu::GpuMat d__simplifiedFlowY(d__previousFrameGray.size(), CV_32FC1);
            cv::gpu::polarToCart(d__simplifiedFlowMagnitude, d__simplifiedFlowAngle, d__simplifiedFlowX, d__simplifiedFlowY, true);

            // initialize DepthMap variables
//            cv::gpu::GpuMat d__depthMap(d__previousFrameGray.size(), CV_32FC1, cv::Scalar(1.0f));
//            depthMapModule.calculate(d__currentFrameGray,d__currentFrameRGB, d__simplifiedFlowX, d__simplifiedFlowY, IMAGEBLOCKSIZE, d__depthMap);
//            cv::Mat depthMat;
//            d__depthMap.download(depthMat);
//            visualizeModule.showMask(depthMat, "Depthmap");
//            #ifdef _TESTING_
//                cv::gpu::GpuMat d__groundTruthDepthMap(d__previousFrameGray.size(), CV_32FC1);
//                getDepth(counter, numberOfImagesBeingProcessed, d__groundTruthDepthMap);
//                tester.testDepth(d__depthMap, d__groundTruthDepthMap, temp_sum);
//                testSum += temp_sum;
//            #endif

            // initialize 3DFlow variables
//            cv::gpu::GpuMat d__flow3DAngle(d__previousFrameGray.size(), CV_32FC1);
//            cv::gpu::GpuMat d__flow3DMagnitude(d__previousFrameGray.size(), CV_32FC1);
//            flowVector3DModule.calculate(d__simplifiedFlowX, d__simplifiedFlowY, d__depthMap, d__flow3DAngle, d__flow3DMagnitude);
//            visualizeModule.showFlow(d__flow3DMagnitude, d__flow3DAngle, "3D flow");
//            d__depthMap.release();

            // initialize global motion variables
//            cv::gpu::GpuMat d__globalMotionX(d__previousFrameGray.size(), CV_32FC1);
//            cv::gpu::GpuMat d__globalMotionY(d__previousFrameGray.size(), CV_32FC1);
//            cv::gpu::GpuMat d__flow3DX, d__flow3DY;
//            cv::gpu::polarToCart(d__flow3DMagnitude, d__flow3DAngle, d__flow3DX, d__flow3DY, true);
//            globalMotion.calculate(d__flow3DX, d__flow3DY,100, 5, 0.0000001f, 5, d__globalMotionX, d__globalMotionY);
//            cv::gpu::GpuMat d__globalMotionMagnitude, d__globalMotionAngle;
//            cv::gpu::cartToPolar(d__globalMotionX, d__globalMotionY, d__globalMotionMagnitude, d__globalMotionAngle, true);
//            visualizeModule.showFlow(d__globalMotionMagnitude, d__globalMotionAngle, "Globalmotion");
//            #ifdef _TESTING_
//                float globalMotionX, globalMotionY, globalMotionAngle;
//                globalMotionX = globalMotionY = globalMotionAngle = -1.0f;
//                globalMotion.getGlobalMotionParameters(globalMotionX, globalMotionY, globalMotionAngle);
//                tester.testGlobalMotion(globalMotionX, globalMotionY, globalMotionAngle, counter, temp_sum);
//                testSum += temp_sum;
//            #endif

            // initialize motion subtraction variables
//            cv::gpu::GpuMat d__subtracktedAngle(d__previousFrameGray.size(), CV_32FC1);
//            cv::gpu::GpuMat d__subtracktedMagnitude(d__previousFrameGray.size(), CV_32FC1);
//            motionSubtraction.subtractGlobalMotion(d__flow3DAngle, d__flow3DMagnitude, d__globalMotionX, d__globalMotionY, d__subtracktedAngle, d__subtracktedMagnitude);
//            visualizeModule.showFlow(d__subtracktedMagnitude, d__subtracktedAngle, "Subtracted Flow");
//            d__flow3DAngle.release();
//            d__flow3DMagnitude.release();
//            d__globalMotionX.release();
//            d__globalMotionY.release();

            // initialize Segmentation variables
            cv::Mat mask(d__previousFrameGray.size(), CV_32FC1);
            cv::gpu::GpuMat d__segments(d__previousFrameGray.size(), CV_32FC1, cv::Scalar(0.0f));
            segmentationModule.segment(d__currentFrameRGB, d__opticalFlowX, d__opticalFlowY, d__segments);

            double min = 0.0;
            double max = 0.0;
            cv::gpu::minMax(d__segments, &min, &max);
            cv::Mat segments;
            segments /= max;
            segments *= 255;
            d__segments.download(segments);
//            cv::cvtColor(segments, segments, )
            cv::imshow("Segments", segments);
//            d__subtracktedAngle.release();
//            d__subtracktedMagnitude.release();

            // show masked regions
//            mask.convertTo(mask, CV_8U);
//            cv::Mat maskedFrame;
//            currentFrameRGB.copyTo(maskedFrame, mask);
//            imshow("masked frame", maskedFrame);
//            #ifdef _TESTING_
//            cv::Mat groundTruthMaskedFrame(d__groundTruthDepthMap);
//                tester.testMask(maskedFrame, groundTruthMaskedFrame, temp_sum);
//                testSum += temp_sum;
//            #endif
//            mask.release();
//            maskedFrame.release();
//            d__opticalFlowY.release();
//            d__opticalFlowX.release();



            //-----------------------------------------------------------------------------------------------------
            // save current frame in previous frame
            //-----------------------------------------------------------------------------------------------------
            d__currentFrameGray.copyTo(d__previousFrameGray);



            //-----------------------------------------------------------------------------------------------------
            // total time
            //-----------------------------------------------------------------------------------------------------
            const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
            std::cout << "Total time : \t" << timeSec << " sec" << std::endl;
            if(cv::waitKey(30) >= 0) {
                isStopedByUser = true;
                break;
            }
            std::cout << "----------------------------------------------------" << std::endl;



            //-----------------------------------------------------------------------------------------------------
            // check if test will be stopped and parameters will be changed
            //-----------------------------------------------------------------------------------------------------
            counter++;
            if (counter == numberOfImagesBeingProcessed) break;

        }

    }
    }catch(cv::Exception & e) {
        std::cout << "Caught cv::Error: " << std::endl;
        std::cout << e.what() << std::endl;
        std::cout << e.msg << std::endl;
        std::cout << e.code << std::endl;
    }

    return 0;
}
