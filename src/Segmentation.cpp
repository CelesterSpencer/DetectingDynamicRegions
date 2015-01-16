#include "Segmentation.h"

//----------------------------------------------------------------------------------------
// PUBLIC METHODS
//----------------------------------------------------------------------------------------

void Segmentation::calculate(cv::gpu::GpuMat &in__currentFrame, cv::gpu::GpuMat &in__flowVector3DAngle, cv::gpu::GpuMat &in__flowVector3DMagnitude, cv::gpu::GpuMat &out__subtractedAngle, cv::gpu::GpuMat &out__subtractedMagnitude, cv::Mat &out__segments, std::vector<Region> &out__dynamicRegions) {
    const int64 start = cv::getTickCount();

    float globalAngle, globalMagnitude;

    // estimate global motion
    calcGlobalMotion(in__flowVector3DAngle, in__flowVector3DMagnitude, globalAngle, globalMagnitude);

    // convert from magnitde/angle to x/y
    cv::gpu::GpuMat subtractedX, subtractedY, flowVector3DX, flowVector3DY;
    float globalX, globalY;
    cv::Mat temp_x, temp_y;
    cv::Mat temp_magnitude(1, 1, CV_32FC1, cv::Scalar(globalMagnitude));
    cv::Mat temp_angle(1, 1, CV_32FC1, cv::Scalar(globalAngle));
    cv::gpu::polarToCart(in__flowVector3DMagnitude, in__flowVector3DAngle, flowVector3DX, flowVector3DY, true);
    cv::polarToCart(temp_magnitude, temp_angle, temp_x, temp_y, true);
    globalX = temp_x.at<float>(0,0);
    globalY = temp_y.at<float>(0,0);

    // subtract global motion from 3D flow
    subtractGlobalMotion(flowVector3DX, flowVector3DY, globalX, globalY, subtractedX, subtractedY);

    // convert from x/y to magnitde/angle
    cv::gpu::cartToPolar(subtractedX, subtractedY, out__subtractedMagnitude, out__subtractedAngle, true);

    // segmentation
    segmentDynamicObjects(out__subtractedAngle, out__subtractedMagnitude, out__segments);

    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Segmentation : \t" << timeSec << " sec" << std::endl;
}



//----------------------------------------------------------------------------------------
// PRIVATE METHODS
//----------------------------------------------------------------------------------------

void Segmentation::calcGlobalMotion(cv::gpu::GpuMat &in__flowVector3DAngle, cv::gpu::GpuMat &in__flowVector3DMagnitude, float &out__globalAngle, float &out__globalMagnitude) {
    const int64 start = cv::getTickCount();

    // class for segmentation
    SegmentationKernel kernel;
    kernel.setThreadSize(m__threadSize);

    int numberOfBins = m__numberOfAngles * m__numberOfMagnitudes;

    // calculate blocksize
    int size = in__flowVector3DAngle.rows * in__flowVector3DAngle.cols;
    int cols = in__flowVector3DAngle.cols;
    int rows = in__flowVector3DAngle.rows;

    // allocate array on device
    int *d__bins;
    int *binsPtr = new int[size * numberOfBins];
    for(int ind = 0; ind < size * numberOfBins; ind++) {
       *(binsPtr + ind) = 0;
    }
    cudaMalloc((void **)&d__bins, sizeof(int) * size * numberOfBins);
    cudaMemcpy(d__bins, binsPtr, sizeof(int) * size * numberOfBins, cudaMemcpyHostToDevice);

    // copy min and max to gpu
    double minData = 0.0;
    double *ptr_min = &minData;
    double maxData = 0.0;
    double *ptr_max = &maxData;

    // get longest magnitude in d__magnitude and calculate the lengthPerMagnitude
    cv::gpu::minMax(in__flowVector3DMagnitude, ptr_min, ptr_max);
//    printf("biggest value is %f \n", *ptr_max);
    m__lengthPerMagnitude = (float)(*ptr_max);

    // calculate corresponding bin for every entry in d__angle

    kernel.fillBins(in__flowVector3DMagnitude.ptr<float>(), in__flowVector3DAngle.ptr<float>(), in__flowVector3DMagnitude.step, in__flowVector3DAngle.step, cols, rows, m__numberOfMagnitudes,m__numberOfAngles, m__lengthPerMagnitude, d__bins);


    // collect results iteratively
    int tempSize = size;
    while(tempSize > 1) {
       bool isOdd = (tempSize % 2 == 1);
       kernel.sumUpBins(tempSize, isOdd, numberOfBins, d__bins);
       tempSize = tempSize / 2;
    }

    // copy results back
    cudaMemcpy(binsPtr, d__bins, sizeof(int) * size * numberOfBins, cudaMemcpyDeviceToHost);

    // for checking if all pixels were collected
    int resultNumberOfPixel = 0;

    // array with percentage of pixels of the angles
    float *relativeAngleDistribution;
    relativeAngleDistribution = (float*)malloc(sizeof(float) * m__numberOfAngles);

    // array that counts the pixels in every magnitude bin
    float *magnitudePixels;
    magnitudePixels = (float*)malloc(sizeof(float) * m__numberOfMagnitudes);

    // angle with biggest percentage
    float biggestPercentageOfPixels = 0.0;
    int ind_biggestAngle = 0;



    // calculate percentage of each direction
    for(int ind_angle = 0; ind_angle < m__numberOfAngles; ind_angle++) {
       // collect all magnitudes and sub up for every angle
       int numberOfPixelsInAngle = 0;
       for (int ind_magnitude = 0; ind_magnitude < m__numberOfMagnitudes; ind_magnitude++) {
           float numberOfPixelsInBin = binsPtr[ind_angle*m__numberOfMagnitudes + ind_magnitude];
           numberOfPixelsInAngle += numberOfPixelsInBin;
       }
       // calculate percentage of all angles
       relativeAngleDistribution[ind_angle] = (float)numberOfPixelsInAngle / size;
       resultNumberOfPixel += numberOfPixelsInAngle;
       if (relativeAngleDistribution[ind_angle] > biggestPercentageOfPixels) {
            biggestPercentageOfPixels = relativeAngleDistribution[ind_angle];
            ind_biggestAngle = ind_angle;
       }
//       printf("Bin %d : %f \n", ind, relativeNumber[ind]);
    }
    // just checking if all pixels had been landed in a bin
    if (resultNumberOfPixel == size) {
        printf("Size of binvalues and size of pixel matches \n");
    }else {
        printf("Size of binvalues and size of pixel does not match: %d %d \n", resultNumberOfPixel, size);
    }



    // get global motion candidates
    int degreePerBin = 360 / numberOfBins;
    float globalMotionCandidateMagnitude;
    float globalMotionCandidateAngle;



    // most of the objects move across the same angle
    if (biggestPercentageOfPixels > 0.5) {

       printf("Most vectors point to the same direction \n");

       // just average all magnitudes since this direction must be the right direction
       float meanMagnitude = 0;
       int sumOfPixels = 0;
       for (int ind_magnitude = 0; ind_magnitude < m__numberOfMagnitudes; ind_magnitude++) {
            int numberOfPixels = binsPtr[ind_biggestAngle * m__numberOfMagnitudes + ind_magnitude];
            /*
             * we need to calculate ind+1 since we want the biggest bin to contain the maxMagnitude
             * but also the smallest bin is expected to cover null vectors
             */
            float magnitudeLength = (ind_magnitude > 0) ? (ind_magnitude+1) * m__lengthPerMagnitude : 0;
//            printf("magnitude length is %f %d \n", magnitudeLength, numberOfPixels);
            meanMagnitude += (magnitudeLength * numberOfPixels);
            sumOfPixels += numberOfPixels;
       }
       meanMagnitude /= sumOfPixels;

       // store results in globalMotionCandidate
       globalMotionCandidateMagnitude = meanMagnitude;
       globalMotionCandidateAngle = degreePerBin * ind_biggestAngle;



    // select all possible values and add them together
    }else {
       printf("vectors point to several directions \n");

       float resultingXDirection = 0;
       float resultingYDirection = 0;
       cv::Mat temp_angleMat(1,1, CV_32FC1, cv::Scalar(0));
       cv::Mat temp_magnitudeMat(1,1, CV_32FC1, cv::Scalar(0));
       cv::Mat temp_xMat, temp_yMat;

       // wight all direction with respect to their distribution
       for(int ind_angle = 0 ; ind_angle < m__numberOfAngles; ind_angle++) {
            float angle = ind_angle * degreePerBin;
            float weight = relativeAngleDistribution[ind_angle];

            // get mean magnitude
            float meanMagnitude = 0;
            int sumOfPixels = 0;
            for (int ind_magnitude = 0; ind_magnitude < m__numberOfMagnitudes; ind_magnitude++) {
                 int numberOfPixels = binsPtr[ind_biggestAngle * m__numberOfMagnitudes + ind_magnitude];
                 /*
                  * we need to calculate ind+1 since we want the biggest bin to contain the maxMagnitude
                  * but also the smallest bin is expected to cover null vectors
                  */
                 float magnitudeLength = (ind_magnitude > 0) ? (ind_magnitude+1) * m__lengthPerMagnitude : 0;
                 meanMagnitude += (magnitudeLength * numberOfPixels);
                 sumOfPixels += numberOfPixels;
            }
            meanMagnitude /= sumOfPixels;

            // calculate resulting direction
            temp_angleMat.at<float>(0,0) = angle;
            temp_magnitudeMat.at<float>(0,0) = meanMagnitude;
            cv::polarToCart(temp_magnitudeMat, temp_angleMat, temp_xMat, temp_yMat, true);
            resultingXDirection += temp_xMat.at<float>(0,0) * weight;
            resultingYDirection += temp_yMat.at<float>(0,0) * weight;
       }
       temp_xMat.at<float>(0,0) = resultingXDirection;
       temp_yMat.at<float>(0,0) = resultingYDirection;
       cv::cartToPolar(temp_xMat, temp_yMat, temp_magnitudeMat, temp_angleMat);

       // store results in globalMotionCandidate
       globalMotionCandidateMagnitude = temp_magnitudeMat.at<float>(0,0);
       globalMotionCandidateAngle = temp_angleMat.at<float>(0,0);
    }

    // out values
    out__globalAngle = globalMotionCandidateAngle;
    out__globalMagnitude = globalMotionCandidateMagnitude;

    printf("globalMagnitude: %f \n", out__globalMagnitude);
    printf("globalAngle: %f \n", out__globalAngle);

    // free memory
    free(binsPtr);

    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Global motion : \t" << timeSec << " sec" << std::endl;
}



void Segmentation::subtractGlobalMotion(cv::gpu::GpuMat &in__flowVector3DX, cv::gpu::GpuMat &in__flowVector3DY, float in__globalX, float in__globalY, cv::gpu::GpuMat &out__subtractedX, cv::gpu::GpuMat &out__subtractedY) {
    const int64 start = cv::getTickCount();

    in__flowVector3DX.copyTo(out__subtractedX);
    in__flowVector3DY.copyTo(out__subtractedY);

    // class for segmentation
    SegmentationKernel kernel;
    kernel.setThreadSize(m__threadSize);

    // subtract global motion
    kernel.globalMotionSubtractionHost(in__flowVector3DX, in__flowVector3DY, in__globalX, in__globalY, out__subtractedX, out__subtractedY);

    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Subtract global motion : \t" << timeSec << " sec" << std::endl;
}



void Segmentation::segmentDynamicObjects(cv::gpu::GpuMat &flowAngleSubtracted, cv::gpu::GpuMat &flowMagnitudeSubtracted, cv::Mat &out__segments) {

    float degreePerBin = 360.0/m__numberOfBins;

    cv::Mat flowMagnitudeSubtractedHost;
    flowAngleSubtracted.download(out__segments);
    flowMagnitudeSubtracted.download(flowMagnitudeSubtractedHost);

    cv::Size angleMatSize = flowAngleSubtracted.size();

    cv::Mat regions(angleMatSize, CV_32FC1, cv::Scalar(0));
    int counter = 1;

    // region labeling
    for(int y = 0; y < out__segments.rows; y++) {
        for(int x = 0; x < out__segments.cols; x++) {

            // set region to 0 if its magnitude is below a threshold
            float magnitude = flowMagnitudeSubtractedHost.at<float>(y,x);
            if (magnitude < 1.5) {
                //printf("value is below threshold \n");
                regions.at<float>(y,x) = 0;
                continue;
            }

            int up = y - 1;
            int left = x - 1;
            // check if upper neighbor has the same angle

            if (up >= 0) {
                float angle, angleUp;
                angle = out__segments.at<float>(y,x);
                angle /= degreePerBin;
                angleUp = out__segments.at<float>(up,x);
                angleUp /= degreePerBin;
                if (angle == angleUp) {
                    float regionVal = regions.at<float>(up,x);
                    if(regionVal != 0) {
                        regions.at<float>(y,x) = regionVal;
                        continue;
                    }
                }
            }

            // check if left neighbor has the same angle
            if (left >= 0) {
                float angle, angleLeft;
                angle = out__segments.at<float>(y,x);
                angle /= degreePerBin;
                angleLeft = out__segments.at<float>(y,left);
                angleLeft /= degreePerBin;
                if (angle == angleLeft) {
                    float regionVal = regions.at<float>(y,left);
                    if(regionVal != 0) {
                        regions.at<float>(y,x) = regionVal;
                        continue;
                    }
                }
            }
            // in case both neighbors have a different angle assign new label
            regions.at<float>(y,x) = counter;
            counter++;
        }
    }

    // scale labels from 0..1 to 0..255
//    printf("highest counter is %d \n", counter);
    for(int y = 0; y < out__segments.rows; y++) {
        for(int x = 0; x < out__segments.cols; x++) {
            out__segments.at<float>(y,x) = regions.at<float>(y,x) /counter;
            out__segments.at<float>(y,x) *= 255;
        }
    }

}
