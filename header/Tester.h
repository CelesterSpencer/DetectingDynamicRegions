#ifndef _TESTER_H_
#define _TESTER_H_

#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <fstream>
#include <iostream>

/*!
 * \class Responsible for Evaluating the results of this project
 * \author Adrian Derstroff
 * \date 04.02.2014
 */
class Tester {
public:
 Tester() {
    file.open("./res/TestLog.txt");
 }

 ~Tester() {
    file.close();
 }

 void printFrameNumber(int frameNumber) {
    file << "--------------- Frame " << frameNumber << " --------------- \n";
 }

 void readGroundTruthData(std::string pathToFile, int numberOfFrames);
 void testDepth(cv::gpu::GpuMat &testDepth, cv::gpu::GpuMat &groundTruthMap, float &differenceValue);
 void testGlobalMotion(float testX, float testY, float testRotationAngle, int frameNumber, float &differenceValue);
 void testObjectsMotion(float* testObjectsTranslations, int frameNumber, float &differenceValue);
 void testMask(cv::Mat testMask, cv::Mat groundTruthMask, float &differenceValue);
 void startTime();
 void stopTime();
 void resetOverallTime();
 float getOverallTime();

private:
    float* cameraTranslation;
    float* objectsTranslation;
    float overallTime;
    int64 start;
    std::ofstream file;
};

#endif
