#include "Tester.h"

void Tester::readGroundTruthData(std::string pathToFile, int numberOfFrames){

    // set up arrays
    cameraTranslation = new float[numberOfFrames * 2];
    objectsTranslation = new float[numberOfFrames * 2 * 3]; // number of frames * x,y * number of objects

    // read input data
    std::cout << "reading ground truth" << std::endl;
    std::ifstream stream;
    stream.open(pathToFile);
    std::string line;
    if (!stream) {
        std::cerr << "Could not open the file!" << std::endl;
    }else {
        int num = 1;
        while (getline(stream,line)) {
            std::cout << "Frame " << num << std::endl;
            getline(stream,line);
            getline(stream,line);
            int position = line.find(',');
            std::string x = line.substr(0, position);
            std::string y = line.substr(position + 1);
            position = y.find(',');
            std::string angle = y.substr(position + 1);
            y = y.substr(0, position);
            std::cout << x << ", " << y << ", " << angle << '\n';
            getline(stream,line);
            for (int i = 0; i < 3; i++) {
                getline(stream,line);
                int position = line.find(',');
                x = line.substr(0, position);
                y = line.substr(position + 1);
                y.find(',');
                angle = y.substr(position + 1);
                y = y.substr(0, position);
                std::cout << "Object " << i+1 << ": " << x << ", " << y << ", " << angle << '\n';
                cameraTranslation[(num * 3)] = stof(x);
                cameraTranslation[(num * 3) + 1] = stof(y);
                cameraTranslation[(num * 3) + 2] = stof(angle);
            }
            getline(stream,line);
            num++;
            if (num == numberOfFrames) break;
        }
        stream.close();
    }
    std::cout << "--------------------" << std::endl;
}

void Tester::testDepth(cv::gpu::GpuMat &testDepth, cv::gpu::GpuMat &groundTruthMap, float &differenceValue) {

    // download depthmaps
    cv::Mat depth, groundTruth;
    testDepth.download(depth);
    groundTruthMap.download(groundTruth);

    float sum = 0.0f;

    for (int y = 0; y < depth.rows; y++) {
        for (int x = 0; x < depth.cols; x++) {
            sum = (groundTruth.at<float>(y,x) - depth.at<float>(y,x)) * (groundTruth.at<float>(y,x) - depth.at<float>(y,x));
        }
    }

    differenceValue = sum;
    file << "Depth error: " << differenceValue << "\n";

}

void Tester::testGlobalMotion(float testX, float testY, float testRotationAngle, int frameNumber, float &differenceValue) {

    float sum = 0.0;
    sum += (testX - cameraTranslation[frameNumber * 3]) * (testX - cameraTranslation[frameNumber * 3]);                                     // x
    sum += (testY - cameraTranslation[(frameNumber * 3) + 1]) * (testY - cameraTranslation[(frameNumber * 3) + 1]);                           // y
    sum += (testRotationAngle - cameraTranslation[(frameNumber * 3) + 2]) * (testRotationAngle - cameraTranslation[(frameNumber * 3) + 2]);   // angle
    differenceValue = sum;
    file << "Ground truth: " << "\n";
    file << "   Global motion in x: " << cameraTranslation[frameNumber * 3] << "\n";
    file << "   Global motion in y: " << cameraTranslation[(frameNumber * 3) + 1] << "\n";
    file << "   Global motion rotation: " << cameraTranslation[(frameNumber * 3) + 2] << "\n";
    file << "Test data: " << "\n";
    file << "   Global motion in x: " << testX << "\n";
    file << "   Global motion in y: " << testY << "\n";
    file << "   Global motion rotation: " << testRotationAngle << "\n";
    file << "Global motion error: " << differenceValue << "\n";

}

void Tester::testObjectsMotion(float* testObjectsTranslations, int frameNumber, float &differenceValue) {

    float smallestSum = 1000000;
    for (int i = 0; i < 3; i++) {
        float sum = 0.0;
        sum += (testObjectsTranslations[i * 2] - cameraTranslation[frameNumber * 6 + i * 2]) * (testObjectsTranslations[i * 2] - cameraTranslation[frameNumber * 6 + i * 2]);
        sum += (testObjectsTranslations[i * 2 + 1] - cameraTranslation[frameNumber * 6 + i * 2 + 1]) * (testObjectsTranslations[i * 2 + 1] - cameraTranslation[frameNumber * 6 + i * 2 + 1]);
        differenceValue = sum;
        if (smallestSum > sum) smallestSum = sum;
    }
    differenceValue = smallestSum;
    file << "Object motion error: " << differenceValue << "\n";

}

void Tester::testMask(cv::Mat testMask, cv::Mat groundTruthMask, float &differenceValue) {

    int64 numOfWrongPixels = 0;
    int64 numberOfAllPixels = testMask.rows * testMask.cols;

    for (int y = 0; y < testMask.rows; y++) {
        for (int x = 0; x < testMask.cols; x++) {
            if (groundTruthMask.at<float>(y,x) == 0) {
                if (testMask.at<float>(y,x) != 0) {
                    numOfWrongPixels++;
                }
            }else {
                if (testMask.at<float>(y,x) == 0) {
                    numOfWrongPixels++;
                }
            }
        }
    }

    differenceValue = (float)numOfWrongPixels / numberOfAllPixels;
    file << "Mask error: " << differenceValue << "\n";

}

void Tester::startTime() {
    start = cv::getTickCount();
}

void Tester::stopTime() {
    float timeDelta = (cv::getTickCount() - start) / cv::getTickFrequency();
    overallTime += timeDelta;
}

void Tester::resetOverallTime() {
    overallTime = 0;
}

float Tester::getOverallTime() {
    return overallTime;
}
