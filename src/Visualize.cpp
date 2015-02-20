#include "Visualize.h"

void Visualize::showFlow(cv::gpu::GpuMat &flowMagnitude, cv::gpu::GpuMat &flowAngle, std::string windowname) {

    const int64 start = cv::getTickCount();

    cv::gpu::GpuMat _hsv[3], hsv;
    cv::Mat onesMat = cv::Mat::ones(flowAngle.size(), CV_32F);
    cv::gpu::GpuMat onesGpu(onesMat);
    _hsv[0] = flowAngle;
    _hsv[1] = flowMagnitude;
    _hsv[2] = onesGpu;
    cv::gpu::merge(_hsv, 3, hsv);

    //convert to BGR
    cv::gpu::GpuMat bgr;
    cv::gpu::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    //transfer from device to host
    cv::Mat out;
    bgr.download(out);

    // display frame
    imshow(windowname, out);
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
//    std::cout << "Show 3Dflow : \t" << timeSec << " sec" << std::endl;
}

void Visualize::showDepth(float *inptr__depth, int cols, int rows, int numberOfImageBlocks, int blockSize) {

    const int64 start = cv::getTickCount();

    cv::Mat depthMat(rows, cols, CV_32FC1);

    int imageBlocksInX = cols/ blockSize;
    int imageBlocksInY = rows/ blockSize;

    // iterate over imageBlocks and will Mat with value of corresponding imageblock
    for (int ind__y = 0; ind__y < imageBlocksInY; ind__y++){
        for (int ind__x = 0; ind__x < imageBlocksInX; ind__x++){
            int idx = ind__x + (imageBlocksInX * ind__y);
            if (idx >= numberOfImageBlocks) continue;
            float depth = inptr__depth[idx];
            for(int ind__imgBlockY = 0; ind__imgBlockY < blockSize; ind__imgBlockY++) {
                for(int ind__imgBlockX = 0; ind__imgBlockX < blockSize; ind__imgBlockX++) {
                    if (ind__imgBlockY + ind__y < cols && ind__imgBlockX + ind__x < rows) {
                        depthMat.at<float>(ind__imgBlockY + ind__y, ind__imgBlockX + ind__x) = depth;
                    }
                }
            }
        }
    }

    // get 3 layers for BGR image
    cv::Mat r(rows, cols, CV_32FC1, cv::Scalar(0.0));
    cv::Mat g(rows, cols, CV_32FC1, cv::Scalar(0.0));
    cv::Mat bgrs[3];
    bgrs[0] = depthMat;
    bgrs[1] = g;
    bgrs[2] = r;

    // merge 3 layers to 1
    cv::Mat bgrMerged;
    cv::merge(bgrs,3, bgrMerged);

    // display frame
    imshow("coarseDepthmap", bgrMerged);
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
//    std::cout << "Show imageblocks depth : \t" << timeSec << " sec" << std::endl;

}

void Visualize::showMask(cv::Mat regions, std::string windowName) {

    const int64 start = cv::getTickCount();

    // merge 3 layers to 1
    cv::Mat bgrs[3];
    bgrs[0] = regions;
    bgrs[1] = regions;
    bgrs[2] = regions;
    cv::Mat bgrMerged;
    cv::merge(bgrs,3, bgrMerged);

    // display frame
    imshow(windowName, bgrMerged);

    // show time
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
//    std::cout << "Show regions : \t" << timeSec << " sec" << std::endl;
}

void Visualize::showRegions(std::vector<Region> regions, int cols, int rows, std::string windowName) {

    const int64 start = cv::getTickCount();

    cv::Mat r(rows, cols, CV_32FC1, cv::Scalar(0.0));
    cv::Mat g(rows, cols, CV_32FC1, cv::Scalar(0.0));
    cv::Mat b(rows, cols, CV_32FC1, cv::Scalar(0.0));

    int numberOfRegions = regions.size();
    int sizeOfFrame = cols * rows;
    float colorIncreasePerRegions = (265.0f * 265.0f * 265.0f) / numberOfRegions;
    int counter = 0;
    for (Region region : regions) {
        float value = (float)((int)(counter * colorIncreasePerRegions) % 265);
        int channel = (int)((counter * colorIncreasePerRegions) / 265) % 3;
//        std::cout << "Value is " << value << " in channel " << channel << " and increase is " << colorIncreasePerRegions <<  std::endl;
        cv::Mat* ptr__mat;
        switch(channel) {
        case 0:
            for (cv::Point pixel : region.getAllPixels()) {
//                std::cout << "Region " << counter << " has " << region.getAllPixels().size() << " pixels" << std::endl;
                r.at<float>(pixel.y, pixel.x) = value;
            }
            break;
        case 1:
            for (cv::Point pixel : region.getAllPixels()) {
//                std::cout << "Region " << counter << " has " << region.getAllPixels().size() << " pixels" << std::endl;
                g.at<float>(pixel.y, pixel.x) = value;
            }
            break;
        case 2:
            for (cv::Point pixel : region.getAllPixels()) {
//                std::cout << "Region " << counter << " has " << region.getAllPixels().size() << " pixels" << std::endl;
                b.at<float>(pixel.y, pixel.x) = value;
            }
            break;
        }

        counter++;
    }

    // merge 3 layers to 1
    cv::Mat bgrs[3];
    bgrs[0] = b;
    bgrs[1] = g;
    bgrs[2] = r;
    cv::Mat bgrMerged;
    cv::merge(bgrs,3, bgrMerged);

    // display frame
    imshow(windowName, bgrMerged);

    // show time
    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
}
