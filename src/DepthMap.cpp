#include "DepthMap.h"

//----------------------------------------------------------------------------------------
// PUBLIC METHODS
//----------------------------------------------------------------------------------------

void DepthMap::calculate(cv::gpu::GpuMat &in__inputFrameGray, cv::gpu::GpuMat &in__inputFrameRGB, cv::gpu::GpuMat &in__opticalFlowX, cv::gpu::GpuMat &in__opticalFlowY, int in__imageblockSize, cv::gpu::GpuMat &out__depthMap) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
//    int cols = in__inputFrameGray.cols;
//    int rows = in__inputFrameGray.rows;
//    int imageblockSize = in__imageblockSize;
//    int imageblockRows = rows / imageblockSize;
//    int imageblockCols = cols / imageblockSize;
//    int numberOfImageBlocks = imageblockRows * imageblockCols;
//    float *ptr__depth = new float[numberOfImageBlocks];
//    std::fill_n(ptr__depth, numberOfImageBlocks, 0.0f);
//    float *dptr__depth;
//    cudaMalloc((void**)&dptr__depth, sizeof(float) * numberOfImageBlocks);
//    cudaMemcpy(dptr__depth, ptr__depth, sizeof(float) * numberOfImageBlocks, cudaMemcpyHostToDevice);
//    std::cout << "initialized variables " << numberOfImageBlocks <<  std::endl;


    //---------------------------------------------------------------------------------------------
    // calculate coarse depthmap
    //---------------------------------------------------------------------------------------------
//    calcCoarseDepthMap(in__inputFrameGray, in__inputFrameRGB, in__imageblockSize, numberOfImageBlocks ,0.01f, dptr__depth);


    //---------------------------------------------------------------------------------------------
    // fill depthmap with depth values
    //---------------------------------------------------------------------------------------------
//    kernel.fillDepthMap(out__depthMap, numberOfImageBlocks, imageblockSize, dptr__depth);




    //-----------------------------------------------------------------------------------------------------
    // Get path name
    //-----------------------------------------------------------------------------------------------------
    number = (number + 1) % m_numberOfImagesBeingProcessed;
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

//    for (int y = 0; y < normalizedDepth.rows; y++) {
//        for (int x = 0; x < normalizedDepth.cols; x++) {
//            normalizedDepth.at<float>(y,x) = 1 - normalizedDepth.at<float>(y,x);
//        }
//    }

    out__depthMap.upload(normalizedDepth);





    //---------------------------------------------------------------------------------------------
    // release memory
    //---------------------------------------------------------------------------------------------
//    free(ptr__depth);

}





//----------------------------------------------------------------------------------------
// PRIVATE METHODS
//----------------------------------------------------------------------------------------

// compute image depth
void DepthMap::calcCoarseDepthMap(cv::gpu::GpuMat &in__currentFrameGray, cv::gpu::GpuMat &in__currentFrameRGB, int imageblockSize, int numberOfImageBlocks, float in__threshold, float *outptr__depth) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int cols = in__currentFrameGray.cols;
    int rows = in__currentFrameGray.rows;
    int imageblockRows = rows / imageblockSize;
    int imageblockCols = cols / imageblockSize;




    //---------------------------------------------------------------------------------------------
    // 1 compute clarity map, mean map and contrast map
    //---------------------------------------------------------------------------------------------
    cv::gpu::GpuMat d__meanMap(in__currentFrameGray.size(), CV_32FC1, cv::Scalar(0.0));
    cv::gpu::GpuMat d__clarityMap(in__currentFrameGray.size(), CV_32FC1, cv::Scalar(0.0));
    cv::gpu::GpuMat d__contrastMap(in__currentFrameGray.size(), CV_32FC1, cv::Scalar(0.0));

    kernel.calculatedMeanAndClarityMap(in__currentFrameGray, d__meanMap, d__clarityMap);
    std::string name = "show clarity";
    drawGpuMat(d__meanMap, name);
    kernel.calculateContrastMap(in__currentFrameGray, d__meanMap, d__contrastMap);
    name = "show contrast";
    drawGpuMat(d__contrastMap, name);

    d__meanMap.release();



    //---------------------------------------------------------------------------------------------
    // 2 calculate clarity and contrast for every imageblock
    //---------------------------------------------------------------------------------------------
    float* ptr__clarities = new float[numberOfImageBlocks];
    std::fill_n(ptr__clarities, numberOfImageBlocks, 0.0f);
    float* ptr__contrasts = new float[numberOfImageBlocks];
    std::fill_n(ptr__contrasts, numberOfImageBlocks, 0.0f);
    float *dptr__clarities;
    float *dptr__contrasts;
    cudaMalloc((void**)&dptr__clarities, sizeof(float) * numberOfImageBlocks);
    cudaMalloc((void**)&dptr__contrasts, sizeof(float) * numberOfImageBlocks);
    cudaMemcpy(dptr__clarities, ptr__clarities, sizeof(float) * numberOfImageBlocks, cudaMemcpyHostToDevice);
    cudaMemcpy(dptr__contrasts, ptr__contrasts, sizeof(float) * numberOfImageBlocks, cudaMemcpyHostToDevice);

    kernel.calculateClarityAndContrastPerImageblock(d__clarityMap, d__contrastMap, numberOfImageBlocks, imageblockSize, dptr__clarities, dptr__contrasts);

    d__clarityMap.release();
    d__contrastMap.release();



    //---------------------------------------------------------------------------------------------
    // 3 find max clarity and contrast
    //---------------------------------------------------------------------------------------------
    // TODO those values are never used ?????
//    float maxClarity = 0.0;
//    float maxContrast = 0.0;

//    kernel.getMaxClarityAndContrast(dptr__clarities, dptr__contrasts, numberOfImageBlocks, maxClarity, maxContrast);



    //---------------------------------------------------------------------------------------------
    // 4 compute depth for every imageblock
    //---------------------------------------------------------------------------------------------
    float* ptr__depths = new float[numberOfImageBlocks];
    std::fill_n(ptr__depths, numberOfImageBlocks, 0.0f);
    float*dptr__depths;
    cudaMalloc((void**) &dptr__depths, sizeof(float) * numberOfImageBlocks);
    cudaMemcpy(dptr__depths, ptr__depths, sizeof(float) * numberOfImageBlocks, cudaMemcpyHostToDevice);

    kernel.calculateDepthPerImageblock(dptr__clarities, dptr__contrasts, numberOfImageBlocks, dptr__depths);

    cudaFree(dptr__clarities);
    cudaFree(dptr__contrasts);
    free(ptr__clarities);
    free(ptr__contrasts);



    //---------------------------------------------------------------------------------------------
    // 5 convert from RGB to YCrCb
    //---------------------------------------------------------------------------------------------
    cv::gpu::GpuMat d__currentFrameYCbCr(in__currentFrameRGB.size(), CV_32FC3);
    cv::gpu::cvtColor(in__currentFrameRGB, d__currentFrameYCbCr, CV_RGB2YCrCb);
    cv::gpu::GpuMat splittedYCbCrMaps[3];
    cv::gpu::split(in__currentFrameRGB, splittedYCbCrMaps);
    cv::gpu::GpuMat d__YChannel = splittedYCbCrMaps[0];
    cv::gpu::GpuMat d__CbChannel = splittedYCbCrMaps[1];
    cv::gpu::GpuMat d__CrChannel = splittedYCbCrMaps[2];

    d__currentFrameYCbCr.release();



    //---------------------------------------------------------------------------------------------
    // 6 calculate mean Cb and Cr for every imageblock
    //---------------------------------------------------------------------------------------------
    float* ptr__meanCb = new float[numberOfImageBlocks];
    std::fill_n(ptr__meanCb, numberOfImageBlocks, 0.0f);
    float* ptr__meanCr = new float[numberOfImageBlocks];
    std::fill_n(ptr__meanCr, numberOfImageBlocks, 0.0f);
    float *dptr__meanCb;
    float *dptr__meanCr;
    cudaMalloc((void**)&dptr__meanCb, sizeof(float) * numberOfImageBlocks);
    cudaMalloc((void**)&dptr__meanCr, sizeof(float) * numberOfImageBlocks);
    cudaMemcpy(dptr__meanCb, ptr__meanCb, sizeof(float) * numberOfImageBlocks, cudaMemcpyHostToDevice);
    cudaMemcpy(dptr__meanCr, ptr__meanCr, sizeof(float) * numberOfImageBlocks, cudaMemcpyHostToDevice);

    kernel.calculateMeanCbAndCrValues(d__CbChannel, d__CrChannel, numberOfImageBlocks, imageblockSize, dptr__meanCb, dptr__meanCr);

    d__YChannel.release();
    d__CbChannel.release();
    d__CrChannel.release();



    //---------------------------------------------------------------------------------------------
    // 7 imageblocks with similar CrCb values are likely to belong to the same object
    //---------------------------------------------------------------------------------------------
    float* ptr__neighbors = new float[numberOfImageBlocks];
    std::fill_n(ptr__neighbors, numberOfImageBlocks, -1.0f); // -1 means that this array has no neighbors
    cudaMemcpy(ptr__meanCb, dptr__meanCb, sizeof(float) * numberOfImageBlocks, cudaMemcpyDeviceToHost);
    cudaMemcpy(ptr__meanCr, dptr__meanCr, sizeof(float) * numberOfImageBlocks, cudaMemcpyDeviceToHost);

    kernel.mergeNeighborImageblocks(ptr__meanCb, ptr__meanCr, numberOfImageBlocks, imageblockCols, in__threshold, ptr__neighbors);

    float *dptr__neighbors;
    cudaMalloc((void**)&dptr__neighbors, sizeof(float) * numberOfImageBlocks);
    cudaMemcpy(dptr__neighbors, ptr__neighbors, sizeof(float) * numberOfImageBlocks, cudaMemcpyHostToDevice);

    cudaFree(dptr__meanCb);
    cudaFree(dptr__meanCr);
    free(ptr__meanCb);
    free(ptr__meanCr);



    //---------------------------------------------------------------------------------------------
    // 8 for every possible idx collect all imageblocks that belong to the same object and adjust depth and calculate distancelevel
    //---------------------------------------------------------------------------------------------
    float* ptr__adjustedDepth = new float[numberOfImageBlocks];
    std::fill_n(ptr__adjustedDepth, numberOfImageBlocks, 0.0f);
    int* ptr__distancelevel = new int[numberOfImageBlocks];
    std::fill_n(ptr__distancelevel, numberOfImageBlocks, 0.0f);
    float *dptr__adjustedDepth;
    int *dptr__distanceLevel;
    cudaMalloc((void**)&dptr__adjustedDepth, sizeof(float) * numberOfImageBlocks);
    cudaMalloc((void**)&dptr__distanceLevel, sizeof(int) * numberOfImageBlocks);
    cudaMemcpy(dptr__adjustedDepth, ptr__adjustedDepth, sizeof(float) * numberOfImageBlocks, cudaMemcpyHostToDevice);
    cudaMemcpy(dptr__distanceLevel, ptr__distancelevel, sizeof(int) * numberOfImageBlocks, cudaMemcpyHostToDevice);

    kernel.adjustDepthAndDistancelevel(dptr__neighbors, dptr__depths, numberOfImageBlocks, dptr__adjustedDepth, dptr__distanceLevel);

    cudaFree(dptr__depths);
    cudaFree(dptr__neighbors);
    free(ptr__depths);
    free(ptr__neighbors);


    //---------------------------------------------------------------------------------------------
    // normalize depth
    //---------------------------------------------------------------------------------------------
    float *ptr__maxDepth = new float;
    *ptr__maxDepth = 0.0f;
    float *dptr__maxDepth;
    cudaMalloc((void**)&dptr__maxDepth, sizeof(float));
    cudaMemcpy(dptr__maxDepth, ptr__maxDepth, sizeof(float), cudaMemcpyHostToDevice);

    kernel.getMaxDepth(dptr__adjustedDepth, numberOfImageBlocks, dptr__maxDepth);

    kernel.normalizeDepth(dptr__adjustedDepth, dptr__maxDepth, numberOfImageBlocks);


    //---------------------------------------------------------------------------------------------
    // copy results
    //---------------------------------------------------------------------------------------------
    cudaMemcpy(outptr__depth, dptr__adjustedDepth, sizeof(float) * numberOfImageBlocks, cudaMemcpyDeviceToDevice);




    //---------------------------------------------------------------------------------------------
    // release memory
    //---------------------------------------------------------------------------------------------
    cudaFree(dptr__adjustedDepth);
    cudaFree(dptr__distanceLevel);
    cudaFree(dptr__maxDepth);
    free(ptr__adjustedDepth);
    free(ptr__distancelevel);
    free(ptr__maxDepth);

}

void calcFineDepthMap() {

}

void DepthMap::drawGpuMat(cv::gpu::GpuMat d__mat, std::string name) {
    cv::Mat mat;
    d__mat.download(mat);
    m__visualizeModule.showMask(mat, name);
}

void DepthMap::printArray(float *dptr__array, int size, std::string name) {
    float *temp__array = new float[size];
    cudaMemcpy(temp__array, dptr__array, sizeof(float) * size, cudaMemcpyDeviceToHost);
    std::cout << name << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << i << ": " << temp__array[i] << std::endl;
    }
    std::cout << "-----------------------------------------------------" << std::endl;
}
