#include "GlobalMotion.h"

void GlobalMotion::calculate(cv::gpu::GpuMat &in__3DFlowX, cv::gpu::GpuMat &in__3DFlowY, int w1, int w2, float threshold, int coarseLevel, cv::gpu::GpuMat &out__globalMotionX, cv::gpu::GpuMat &out__globalMotionY) {

    const int64 start = cv::getTickCount();

    //---------------------------------------------------------------------------------------------
    // create coarse optical flow field
    //---------------------------------------------------------------------------------------------
    cv::gpu::GpuMat d__coarse3DFlowX(in__3DFlowX.rows / coarseLevel, in__3DFlowX.cols / coarseLevel, CV_32FC1);
    cv::gpu::GpuMat d__coarse3DFlowY(in__3DFlowX.rows / coarseLevel, in__3DFlowX.cols / coarseLevel, CV_32FC1);
    kernel.createCoarse3DFlow(in__3DFlowX, in__3DFlowY, coarseLevel, d__coarse3DFlowX, d__coarse3DFlowY);


    //---------------------------------------------------------------------------------------------
    // apply matched filter to coarse optical flow field
    //---------------------------------------------------------------------------------------------
    cv::gpu::GpuMat d__coarseResultSSDs(d__coarse3DFlowX.size(), CV_32FC1, cv::Scalar(0.0f));
    kernel.calculateSSDs(d__coarse3DFlowX, d__coarse3DFlowY, 0, 0, d__coarse3DFlowX.cols - 1, d__coarse3DFlowX.rows - 1, w1, threshold, d__coarseResultSSDs);



    //---------------------------------------------------------------------------------------------
    // find coarse location of minimal SSD
    //---------------------------------------------------------------------------------------------
    int coarseMinX = 0;
    int coarseMinY = 0;
    kernel.getPositionOfMinSSD(d__coarseResultSSDs, coarseMinX, coarseMinY);
//    drawFOE(d__coarse3DFlowX, d__coarse3DFlowY, coarseMinX, coarseMinY);


    //---------------------------------------------------------------------------------------------
    // get search region
    //---------------------------------------------------------------------------------------------
    int centerX = coarseMinX * coarseLevel + (coarseLevel / 2);
    int centerY = coarseMinY * coarseLevel + (coarseLevel / 2);
    int startX = centerX - w1;
    int startY = centerY - w1;
    if (startX < 0) startX = 0;
    if (startY < 0) startY = 0;
    int endX = centerX + w1;
    int endY = centerY + w1;
    if (endX >= in__3DFlowX.cols) endX = in__3DFlowX.cols - 1;
    if (endY >= in__3DFlowX.rows) endY = in__3DFlowX.rows - 1;
    int searchSpaceSizeX = (endX - startX) + 1;
    int searchSpaceSizeY = (endY - startY) + 1;


    //---------------------------------------------------------------------------------------------
    // apply matched filter to find more precise location of FOE
    //---------------------------------------------------------------------------------------------
    cv::gpu::GpuMat d__resultSSDs(searchSpaceSizeY, searchSpaceSizeX, CV_32FC1, cv::Scalar(-1.0f));
    kernel.calculateSSDs(in__3DFlowX, in__3DFlowY, startX, startY, endX, endY, w2, threshold, d__resultSSDs);
    cv::Mat resultSSD;
    d__resultSSDs.download(resultSSD);


    //---------------------------------------------------------------------------------------------
    // find precise location of minimal SSD
    //---------------------------------------------------------------------------------------------
    int fieldOfExpansionX = 0;
    int fieldOfExpansionY = 0;
    kernel.getPositionOfMinSSD(d__resultSSDs, fieldOfExpansionX, fieldOfExpansionY);
    fieldOfExpansionX += startX;
    fieldOfExpansionY += startY;
    drawFOE(in__3DFlowX, in__3DFlowY, fieldOfExpansionX, fieldOfExpansionY, startX, startY, endX, endY);


    //---------------------------------------------------------------------------------------------
    // find biggest and smallest flow vector
    //---------------------------------------------------------------------------------------------
    double *ptr__minFlow = new double;
    double *ptr__maxFlow = new double;
    std::fill_n(ptr__minFlow, 1, 0.0f);
    std::fill_n(ptr__maxFlow, 1, 0.0f);
    cv::gpu::minMax(in__3DFlowX, ptr__minFlow, ptr__maxFlow);
    float minFlowX = (float)*ptr__minFlow;
    float maxFlowX = (float)*ptr__maxFlow;
    cv::gpu::minMax(in__3DFlowY, ptr__minFlow, ptr__maxFlow);
    float minFlowY = (float)*ptr__minFlow;
    float maxFlowY = (float)*ptr__maxFlow;
    float minFlow = (minFlowX < minFlowY) ? minFlowX : minFlowY;
    float maxFlow = (maxFlowX > maxFlowY) ? maxFlowX : maxFlowY;


    //---------------------------------------------------------------------------------------------
    // create several synthetic flowfields
    //---------------------------------------------------------------------------------------------
    std::vector<cv::gpu::GpuMat> syntheticFlowFields;
    float minTranslation = 0;
    float maxTranslation = maxFlow;
    int translationGranularity = 10;    // n + 1 translations (0 and max translation are inculded)
    int directionsGranularity = 4;      // n directions (from 0 to n - 1)
    float maxRotation = maxFlow;
    int rotationGranularity = 1;        // 2n + 1 rotations

    for (int ind__rotation = -rotationGranularity; ind__rotation <= rotationGranularity; ind__rotation++){
        for (int ind__translation = 0; ind__translation <= translationGranularity; ind__translation++) {
            for (int ind__direction = 0; ind__direction < directionsGranularity; ind__direction++) {
                cv::gpu::GpuMat syntheticFlowFieldX(in__3DFlowX.size(), CV_32FC1, cv::Scalar(0.0f));
                cv::gpu::GpuMat syntheticFlowFieldY(in__3DFlowX.size(), CV_32FC1, cv::Scalar(0.0f));

                float interpolation = (ind__translation > 0) ? (float)translationGranularity / ind__translation : 0;
                float translation = interpolation * maxTranslation + (1 - interpolation) * minTranslation;
                float angle = (360.0f / directionsGranularity) * ind__direction;
                float translationY = sin(angle) * translation;
                float translationX = cos(angle) * translation;
                float rotation = (ind__rotation > 0) ? (rotationGranularity / ind__rotation) * maxRotation : 0;

                kernel.createSyntheticFlowField(syntheticFlowFieldX, syntheticFlowFieldY, fieldOfExpansionX, fieldOfExpansionY, rotation, translationX, translationY);

                syntheticFlowFields.push_back(syntheticFlowFieldX);
                syntheticFlowFields.push_back(syntheticFlowFieldY);
            }
        }
    }



    //---------------------------------------------------------------------------------------------
    // compare synthetic flowfields to current observed flowfield
    //---------------------------------------------------------------------------------------------
    int indexOfSmallestDivergence = -1;
    kernel.calculateDivergenceOfFlowFields(syntheticFlowFields, in__3DFlowX, in__3DFlowY, indexOfSmallestDivergence);



    //---------------------------------------------------------------------------------------------
    // get global motion
    //---------------------------------------------------------------------------------------------
    out__globalMotionX = syntheticFlowFields.at(indexOfSmallestDivergence);
    out__globalMotionY = syntheticFlowFields.at(indexOfSmallestDivergence + 1);
    int ind__direction = indexOfSmallestDivergence % directionsGranularity;
    int ind__translation = (indexOfSmallestDivergence / directionsGranularity) % translationGranularity;
    int ind__rotation = (indexOfSmallestDivergence / (directionsGranularity * translationGranularity));

    float interpolation = (ind__translation > 0) ? translationGranularity / ind__translation : 0;
    float translation = interpolation * maxTranslation + (1 - interpolation) * minTranslation;
    float angleOfTranslation = (360.0f / directionsGranularity) * ind__direction;
    m_translationY = sin(angleOfTranslation) * translation;
    m_translationX = cos(angleOfTranslation) * translation;
    m_angle = (ind__rotation > 0) ? (rotationGranularity / ind__rotation) * maxRotation : 0;




    //----------------------------------------------------------------------------------------
    // display computation time
    //----------------------------------------------------------------------------------------
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "Globalmotion : \t" << timeSec << " sec" << std::endl;

}

void GlobalMotion::drawFOE(cv::gpu::GpuMat d__coarse3DFlowX, cv::gpu::GpuMat d__coarse3DFlowY, int centerX, int centerY, int startX, int startY, int endX, int endY ) {

    //---------------------------------------------------------------------------------------------
    // convert to polar
    //---------------------------------------------------------------------------------------------
    cv::gpu::GpuMat coarseFlowMatMagnitude;
    cv::gpu::GpuMat coarseFlowMatAngle;
    cv::gpu::cartToPolar(d__coarse3DFlowX, d__coarse3DFlowY, coarseFlowMatMagnitude, coarseFlowMatAngle, true);

    //---------------------------------------------------------------------------------------------
    // create hsv image
    //---------------------------------------------------------------------------------------------
    cv::gpu::GpuMat _hsv[3], hsv;
    cv::gpu::GpuMat onesGpu(coarseFlowMatMagnitude.size(), CV_32FC1, cv::Scalar(1.0f));
    _hsv[0] = coarseFlowMatAngle;
    _hsv[1] = coarseFlowMatMagnitude;
    _hsv[2] = onesGpu;
    cv::gpu::merge(_hsv, 3, hsv);

    //---------------------------------------------------------------------------------------------
    // convert to bgr
    //---------------------------------------------------------------------------------------------
    cv::gpu::GpuMat bgr;
    cv::gpu::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    //---------------------------------------------------------------------------------------------
    // transfer to host
    //---------------------------------------------------------------------------------------------
    cv::Mat out;
    bgr.download(out);

    //---------------------------------------------------------------------------------------------
    // draw FOE
    //---------------------------------------------------------------------------------------------
    cv::Point center(centerX, centerY);
    int thickness = 1;
    int lineType = 8;
    cv::circle(
         out,
         center,
         5,
         cv::Scalar( 255, 0, 0),
         thickness,
         lineType
    );

    cv::Point start(startX, startY);
    cv::Point end(endX, endY);
    cv::rectangle(out, start, end, cv::Scalar(255, 0, 0), thickness, lineType);

    //---------------------------------------------------------------------------------------------
    // show image
    //---------------------------------------------------------------------------------------------
    cv::imshow("Field of Expansion", out);

}

void GlobalMotion::drawFlow(cv::Mat coarseFlowMatMagnitude, cv::Mat coarseFlowMatAngle, std::string windowName) {

    //---------------------------------------------------------------------------------------------
    // create hsv image
    //---------------------------------------------------------------------------------------------
    cv::Mat _hsv[3], hsv;
    cv::Mat onesGpu(coarseFlowMatMagnitude.size(), CV_32FC1, cv::Scalar(1.0f));
    _hsv[0] = coarseFlowMatAngle;
    _hsv[1] = coarseFlowMatMagnitude;
    _hsv[2] = onesGpu;
    cv::merge(_hsv, 3, hsv);

    //---------------------------------------------------------------------------------------------
    // convert to bgr
    //---------------------------------------------------------------------------------------------
    cv::Mat out;
    cv::cvtColor(hsv, out, cv::COLOR_HSV2BGR);

    //---------------------------------------------------------------------------------------------
    // show image
    //---------------------------------------------------------------------------------------------
    cv::imshow(windowName, out);

}
