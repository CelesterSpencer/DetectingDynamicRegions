#include "Segmentation.h"
#include <unordered_map>

//----------------------------------------------------------------------------------------
// PUBLIC METHODS
//----------------------------------------------------------------------------------------

void Segmentation::calculate(cv::gpu::GpuMat &in__currentFrame, cv::gpu::GpuMat in__subtractedMagnitude, cv::gpu::GpuMat in__subtractedAngle, cv::Mat out__maskedRegions, std::vector<Region> &outptr__regions) {
    const int64 start = cv::getTickCount();

    //---------------------------------------------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------------------------------------------
    float degreePerBin = 360.0/32;
    cv::Mat subtractedFlowMagnitude, subtractedFlowAngle;
    in__subtractedAngle.download(subtractedFlowAngle);
    in__subtractedMagnitude.download(subtractedFlowMagnitude);
    int cols = subtractedFlowAngle.cols;
    int rows = subtractedFlowAngle.rows;
    int counter = 1;
    double *minMagnitude = new double;
    double *maxMagnitude = new double;
    cv::gpu::minMax(in__subtractedMagnitude, minMagnitude, maxMagnitude);
    float threshold = *maxMagnitude * 0.01;

    std::unordered_map<int, Region> regions;

    //---------------------------------------------------------------------------------------------------------------------------------
    // region labeling
    //---------------------------------------------------------------------------------------------------------------------------------
    for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {

            //---------------------------------------------------------------------------------------------------------------------------------
            // get left and upper neighbor
            //---------------------------------------------------------------------------------------------------------------------------------
            float magnitude = subtractedFlowMagnitude.at<float>(y,x);
            int up = y - 1;
            int left = x - 1;

            //---------------------------------------------------------------------------------------------------------------------------------
            // set region to 0 if its magnitude is below a threshold
            //---------------------------------------------------------------------------------------------------------------------------------
            if (magnitude < threshold) {
                out__maskedRegions.at<float>(y,x) = 0;
            }else {
                //---------------------------------------------------------------------------------------------------------------------------------
                // check if upper neighbor has the same angle
                //---------------------------------------------------------------------------------------------------------------------------------
                if (up >= 0) {
                    float angle, angleUp, r, g, b, rUp, gUp, bUp;

                    //
                    angle = subtractedFlowAngle.at<float>(y,x);


                    angleUp = subtractedFlowAngle.at<float>(up,x);

                    if (angle == angleUp) {
                        float regionVal = out__maskedRegions.at<float>(up,x);
                        if(regionVal != 0) {
                            out__maskedRegions.at<float>(y,x) = regionVal;
                            std::unordered_map<int, Region>::const_iterator it = regions.find(regionVal);
                            if (it == regions.end()) {
                                Region region(cols, rows);
                                region.addPixel(x,y);
                                regions.insert(std::pair<float,Region>(regionVal,region));
                            }else {
                                Region region = it->second;
                                region.addPixel(x,y);
                            }
                            continue;
                        }
                    }
                }

                //---------------------------------------------------------------------------------------------------------------------------------
                // check if left neighbor has the same angle
                //---------------------------------------------------------------------------------------------------------------------------------
                if (left >= 0) {
                    float angle, angleLeft;
                    angle = subtractedFlowAngle.at<float>(y,x);
                    angle /= degreePerBin;
                    angleLeft = subtractedFlowAngle.at<float>(y,left);
                    angleLeft /= degreePerBin;
                    if (angle == angleLeft) {
                        float regionVal = out__maskedRegions.at<float>(y,left);
                        if(regionVal != 0) {
                            out__maskedRegions.at<float>(y,x) = regionVal;
                            std::unordered_map<int, Region>::const_iterator it = regions.find(regionVal);
                            if (it == regions.end()) {
                                Region region(cols, rows);
                                region.addPixel(x,y);
                                regions.insert(std::pair<float,Region>(regionVal,region));
                            }else {
                                Region region = it->second;
                                region.addPixel(x,y);
                            }
                            continue;
                        }
                    }
                }

                //---------------------------------------------------------------------------------------------------------------------------------
                // in case both neighbors have a different angle assign new label
                //---------------------------------------------------------------------------------------------------------------------------------
                out__maskedRegions.at<float>(y,x) = counter;
                std::unordered_map<int, Region>::const_iterator it = regions.find(counter);
                if (it == regions.end()) {
                    Region region(cols, rows);
                    region.addPixel(x,y);
                    regions.insert(std::pair<float,Region>(counter,region));
                }else {
                    Region region = it->second;
                    region.addPixel(x,y);
                }
                counter++;
            }
        }
    }

    //----------------------------------------------------------------------------------------
    // push all regions in a vector
    //----------------------------------------------------------------------------------------
    outptr__regions.reserve(regions.size());
    for (auto region : regions) {
        outptr__regions.push_back(region.second);
    }
//    std::cout << "Found " << counter << " different regions" << std::endl;

    //----------------------------------------------------------------------------------------
    // display computation time
    //----------------------------------------------------------------------------------------
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Segmentation : \t" << timeSec << " sec" << std::endl;

}

void Segmentation::segment(cv::gpu::GpuMat &in__frameRGB, cv::gpu::GpuMat &din__flowX, cv::gpu::GpuMat &din__flowY, cv::gpu::GpuMat &out__segments) {

    //----------------------------------------------------------------------------------------
    // convert to YUV color model
    //----------------------------------------------------------------------------------------
    cv::gpu::GpuMat d__currentFrameYUV(in__frameRGB.size(), CV_32FC3);
    cv::gpu::cvtColor(in__frameRGB, d__currentFrameYUV, CV_RGB2YUV);
    cv::gpu::GpuMat splittedYUVMaps[3];
    cv::gpu::split(in__frameRGB, splittedYUVMaps);
    cv::gpu::GpuMat d__YChannel = splittedYUVMaps[0];
    cv::gpu::GpuMat d__UChannel = splittedYUVMaps[1];
    cv::gpu::GpuMat d__VChannel = splittedYUVMaps[2];



    //----------------------------------------------------------------------------------------
    // segment frame depending on flow, color and location cues
    //----------------------------------------------------------------------------------------
    if (!m__isFirstSegmentation) {

        //----------------------------------------------------------------------------------------
        // 0 setup variables
        //----------------------------------------------------------------------------------------
        cv::gpu::GpuMat maxFlowLogLikeliHoods(din__flowX.size(), CV_32FC1, cv::Scalar(0.0));
        cv::gpu::GpuMat sumOfSpatialMeans(din__flowX.size(), CV_32FC1, cv::Scalar(0.0));
        cv::gpu::GpuMat *dptr__flowLogLikeliHoods = new cv::gpu::GpuMat[m__kMax];
        cv::gpu::GpuMat *dptr__colorLogLikeliHoods = new cv::gpu::GpuMat[m__kMax];
        cv::gpu::GpuMat *dptr__likelihoods = new cv::gpu::GpuMat[m__kMax];
        cv::gpu::GpuMat *dptr__covarianzMatrices = new cv::gpu::GpuMat[m__kMax];



        //----------------------------------------------------------------------------------------
        // 1 iterate over all classes and calculate the flow and color likelihood and determine max flow likelihood
        //----------------------------------------------------------------------------------------
        for (int k = 0; k < m__kMax; k++ ) {


            //----------------------------------------------------------------------------------------
            // create binary mask
            //----------------------------------------------------------------------------------------


            //----------------------------------------------------------------------------------------
            // 1.1 calculate mean and covarianz matrix
            //----------------------------------------------------------------------------------------
            float *meanVector = new float[7];
            std::fill_n(meanVector, 7, -1.0f);
            cv::gpu::GpuMat d__covarianzMatrix(7, 7, CV_32FC1, cv::Scalar(0.0f));



            kernel.calcMean(m__classes, d__YChannel, d__UChannel, d__VChannel, din__flowX, din__flowY, k,  m__numberOfPointsPerClass[k], meanVector);
            kernel.calcCovarianzMatrix(m__classes, d__YChannel, d__UChannel, d__VChannel, din__flowX, din__flowY, meanVector, k, m__numberOfPointsPerClass[k], d__covarianzMatrix);
            cv::Mat covMat;
            d__covarianzMatrix.download(covMat);



            //----------------------------------------------------------------------------------------
            // 1.2 calculate flow and color PDF
            //----------------------------------------------------------------------------------------
            cv::gpu::GpuMat d__flowLogLikelihood(din__flowX.size(), CV_32FC1, cv::Scalar(0.0));
            cv::gpu::GpuMat d__colorLogLikelihood(din__flowX.size(), CV_32FC1, cv::Scalar(0.0));
            kernel.calculateFlowAndColorLikelihood(
                        d__YChannel, d__UChannel, d__VChannel,
                        din__flowX, din__flowY,
                        d__covarianzMatrix, meanVector,
                        d__flowLogLikelihood, d__colorLogLikelihood,
                        maxFlowLogLikeliHoods);



            //----------------------------------------------------------------------------------------
            // 1.3 calculate spatial means
            //----------------------------------------------------------------------------------------
            cv::gpu::GpuMat d__binaryImage(din__flowX.size(), CV_32FC1, cv::Scalar(0.0));
            cv::gpu::GpuMat d__gaussianImage(din__flowX.size(), CV_32FC1, cv::Scalar(0.0));
            kernel.makeBinaryImage(m__classes, k, d__binaryImage);
            float sigma = covMat.at<float>(0,1);
            std::cout << "Sigma is " << sigma << std::endl;
            cv::Size2i size((int)31,(int)31);
            cv::Ptr<cv::gpu::FilterEngine_GPU> filter = cv::gpu::createGaussianFilter_GPU(CV_32FC1, size, sigma);
            filter->apply(d__binaryImage,d__gaussianImage);
            kernel.matAdd(d__gaussianImage, sumOfSpatialMeans);




            //----------------------------------------------------------------------------------------
            // 1.4 save temporary results
            //----------------------------------------------------------------------------------------
            dptr__flowLogLikeliHoods[k] = d__flowLogLikelihood.clone();
            dptr__colorLogLikeliHoods[k] = d__colorLogLikelihood.clone();
            dptr__covarianzMatrices[k] = d__covarianzMatrix.clone();



            //----------------------------------------------------------------------------------------
            // 1.5 release memory
            //----------------------------------------------------------------------------------------
            d__flowLogLikelihood.release();
            d__colorLogLikelihood.release();
            d__covarianzMatrix.release();
            d__binaryImage.release();
            d__gaussianImage.release();

        }



        //----------------------------------------------------------------------------------------
        // 2 iterate over all classes and calculate the final likelihood for every class
        //----------------------------------------------------------------------------------------
        for (int k = 0; k < m__kMax; k++ ) {
            cv::gpu::GpuMat d__colorLogLikeLihood = dptr__colorLogLikeliHoods[k];
            cv::gpu::GpuMat d__flowLogLikeLihood = dptr__flowLogLikeliHoods[k];
            cv::gpu::GpuMat d__covarianzMatrix = dptr__covarianzMatrices[k];
            cv::Mat covarianzMatrix;
            d__covarianzMatrix.download(covarianzMatrix);
            cv::gpu::GpuMat likelihood(din__flowX.size(), CV_32FC1, cv::Scalar(0.0));
            int numberOfPoints = m__numberOfPointsPerClass[k];
            float sigma = covarianzMatrix.at<float>(1,0);
            int halfSearchregion = 1;

            kernel.calculateLikelihood(d__colorLogLikeLihood, d__flowLogLikeLihood, sumOfSpatialMeans, maxFlowLogLikeliHoods, numberOfPoints, sigma, halfSearchregion, likelihood);
            dptr__likelihoods[k] = likelihood.clone();

            dptr__colorLogLikeliHoods[k].release();
            dptr__flowLogLikeliHoods[k].release();
            covarianzMatrix.release();
            likelihood.release();
        }



        //----------------------------------------------------------------------------------------
        // 3 for every pixel find the class with the biggest likelihood
        //----------------------------------------------------------------------------------------
        cv::gpu::GpuMat maxLikelihoods(din__flowX.size(), CV_32FC1, cv::Scalar(-1.0));
        cv::gpu::GpuMat maxClasses(din__flowX.size(), CV_32FC1, cv::Scalar(0.0));
        for (int k = 0; k < m__kMax; k++ ) {
            kernel.getBiggestLikelihood(maxLikelihoods, maxClasses, dptr__likelihoods[k], k);
        }



        //----------------------------------------------------------------------------------------
        // 4 calculate number of classes and pixel per class
        //----------------------------------------------------------------------------------------
        double minClass = 0.0;
        double maxClass = 0.0;
        cv::gpu::minMax(maxClasses, &minClass, &maxClass);
        m__kMax = (int)maxClass;
        m__numberOfPointsPerClass.reserve(m__kMax);
        std::cout << "Max class is " << m__kMax << std::endl;
        for (int k = 0; k < m__kMax; k++) {
            cv::gpu::GpuMat d__classK;
            kernel.makeBinaryImage(maxClasses, k, d__classK);
            m__numberOfPointsPerClass[k] = (int)cv::gpu::sum(d__classK)[0];
            std::cout << m__numberOfPointsPerClass[k] << "points in Class " << k << std::endl;
        }



//        //----------------------------------------------------------------------------------------
//        // 4 save results
//        //----------------------------------------------------------------------------------------
        m__classes = maxClasses.clone();
        out__segments = maxClasses.clone();



        //----------------------------------------------------------------------------------------
        // 5 release memory
        //----------------------------------------------------------------------------------------
        maxFlowLogLikeliHoods.release();
        sumOfSpatialMeans.release();
        for (int k = 0; k < m__kMax; k++) {
            dptr__likelihoods[k].release();
        }

    }

    //----------------------------------------------------------------------------------------
    // if first frame calculate an initial segmentation
    //----------------------------------------------------------------------------------------
    else {

        cv::Mat mask(in__frameRGB.size(), CV_32S,  cv::Scalar(0));
        m__classes.upload(mask);
        m__numberOfPointsPerClass.push_back(mask.cols * mask.rows);

        m__isFirstSegmentation = false;

    }

}


