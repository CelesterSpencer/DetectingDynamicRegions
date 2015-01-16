#include "DepthMap.h"

//----------------------------------------------------------------------------------------
// PUBLIC METHODS
//----------------------------------------------------------------------------------------

void DepthMap::calculate(cv::gpu::GpuMat &in__inputFrameGray, cv::gpu::GpuMat &in__inputFrameRGB, cv::gpu::GpuMat &in__opticalFlowX, cv::gpu::GpuMat &in__opticalFlowY, cv::gpu::GpuMat &out__depthMap) {
    out__depthMap = cv::gpu::GpuMat(in__inputFrameGray.size(), CV_32FC1, cv::Scalar(1.0));
    calcCoarseDepthMap(in__inputFrameGray, in__inputFrameRGB, 0.02);
}



//----------------------------------------------------------------------------------------
// PRIVATE METHODS
//----------------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////////////////////
///
/// BUY: "A Real-Time 2-D to 3-D Image Conversion Technique Using Computed Image Depth" pdf
///
//////////////////////////////////////////////////////////////////////////////////////////

// computed image depth
void DepthMap::calcCoarseDepthMap(cv::gpu::GpuMat &in__currentFrameGray, cv::gpu::GpuMat &in__currentFrameRGB, float in__threshold) {

    std::cout << "start calc coarsedepthmap" << std::endl;

    // important variables
    int cols = in__currentFrameGray.cols;
    int rows = in__currentFrameGray.rows;
    int imageblockSize = 16;
    int imageblockRows = rows / imageblockSize;
    int imageblockCols = cols / imageblockSize;
    int numberOfImageBlocks = imageblockRows * imageblockCols;

    std::cout << "initialized variables" << std::endl;



    //---------------------------------------------------------------------------------------------
    // 1 compute clarity map, mean map and contrast map
    //---------------------------------------------------------------------------------------------
    cv::gpu::GpuMat d__meanMap(in__currentFrameGray.size(), CV_32FC1, cv::Scalar(0.0));
    cv::gpu::GpuMat d__clarityMap(in__currentFrameGray.size(), CV_32FC1, cv::Scalar(0.0));
    cv::gpu::GpuMat d__contrastMap(in__currentFrameGray.size(), CV_32FC1, cv::Scalar(0.0));

    kernel.calculatedMeanAndClarityMap(in__currentFrameGray, d__meanMap, d__clarityMap);
    kernel.calculateContrastMap(in__currentFrameGray, d__meanMap, d__contrastMap);

    d__meanMap.release();
    std::cout << "computed clarity map and contrast map"<< std::endl;



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

//    kernel.calculateClarityAndContrastPerImageblock(d__clarityMap, d__contrastMap, numberOfImageBlocks, imageblockSize, dptr__clarities, dptr__contrasts);

    d__clarityMap.release();
    d__contrastMap.release();
    std::cout << "calculated clarity and contrast per imageblock" << std::endl;



    //---------------------------------------------------------------------------------------------
    // 3 find max clarity and contrast
    //---------------------------------------------------------------------------------------------
    // TODO those values are never used ?????
    float maxClarity = 0.0;
    float maxContrast = 0.0;

//    kernel.getMaxClarityAndContrast(dptr__clarities, dptr__contrasts, numberOfImageBlocks, maxClarity, maxContrast);

    std::cout << "got max clarity and contrast" << std::endl;



    //---------------------------------------------------------------------------------------------
    // 4 compute depth for every imageblock
    //---------------------------------------------------------------------------------------------
    float* ptr__depths = new float[numberOfImageBlocks];
    std::fill_n(ptr__depths, numberOfImageBlocks, 0.0f);
    float*dptr__depths;
    cudaMalloc((void**) &dptr__depths, sizeof(float) * numberOfImageBlocks);
    cudaMemcpy(dptr__depths, ptr__depths, sizeof(float) * numberOfImageBlocks, cudaMemcpyHostToDevice);

//    kernel.calculateDepthPerImageblock(dptr__clarities, dptr__contrasts, numberOfImageBlocks, dptr__depths);

    cudaFree(dptr__clarities);
    cudaFree(dptr__contrasts);
    free(ptr__clarities);
    free(ptr__contrasts);
    std::cout << "calculated depth" << std::endl;



    //---------------------------------------------------------------------------------------------
    // 5 convert from RGB to YCrCb
    //---------------------------------------------------------------------------------------------
    cv::gpu::GpuMat d__currentFrameYCbCr(in__currentFrameRGB.size(), CV_32FC3);
    std::cout << "created YCbCr frame" << std::endl;
    cv::gpu::cvtColor(in__currentFrameRGB, d__currentFrameYCbCr, CV_RGB2HSV);
    std::cout << "converted color" << std::endl;
    cv::gpu::GpuMat splittedYCbCrMaps[3];
    cv::gpu::split(in__currentFrameRGB, splittedYCbCrMaps);
    std::cout << "split image" << std::endl;
    cv::gpu::GpuMat d__YChannel = splittedYCbCrMaps[0];
    cv::gpu::GpuMat d__CbChannel = splittedYCbCrMaps[1];
    cv::gpu::GpuMat d__CrChannel = splittedYCbCrMaps[2];


    printFreeSpace();
    std::cout << "converted rgb toycbcr" << std::endl;



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
    std::cout << "calculated mean cb and cr" << std::endl;



    //---------------------------------------------------------------------------------------------
    // 7 imageblocks with similar CrCb values are likely to belong to the same object
    //---------------------------------------------------------------------------------------------
    float* ptr__neighbors = new float[numberOfImageBlocks];
    std::fill_n(ptr__neighbors, numberOfImageBlocks, -1.0f); // -1 means that this array has no neighbors
    float *dptr__neighbors;
    cudaMalloc((void**)&dptr__neighbors, sizeof(float) * numberOfImageBlocks);
    cudaMemcpy(dptr__neighbors, ptr__neighbors, sizeof(float) * numberOfImageBlocks, cudaMemcpyHostToDevice);

//    kernel.mergeNeighborImageblocks(dptr__meanCb, dptr__meanCr, numberOfImageBlocks, imageblockCols, in__threshold, dptr__neighbors);

    cudaFree(dptr__meanCb);
    cudaFree(dptr__meanCr);
    free(ptr__meanCb);
    free(ptr__meanCr);
    std::cout << "found neighbors" << std::endl;



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

//    kernel.adjustDepthAndDistancelevel(dptr__neighbors, dptr__depths, numberOfImageBlocks, dptr__adjustedDepth, dptr__distanceLevel);

    cudaFree(dptr__depths);
    cudaFree(dptr__neighbors);
    free(ptr__depths);
    free(ptr__neighbors);
    std::cout << "adjusted depth and calculated distancelevels" << std::endl;




//     release memory
    cudaFree(dptr__adjustedDepth);
    cudaFree(dptr__distanceLevel);
    free(ptr__adjustedDepth);
    free(ptr__distancelevel);
    printFreeSpace();
    std::cout << "freed memory" << std::endl;

}



void calcFineDepthMap() {

}


