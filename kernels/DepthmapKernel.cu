#include "DepthmapKernel.h"

//------------------------------------------------------------------
// MEAN AND CLARITY
//------------------------------------------------------------------

__global__ void dv__calculatedMeanAndClarityMap(float *inptr__currentFrame, size_t step, int cols, int rows, float *outptr__meanMap, float *outptr__clarityMap) {

    int size = cols * rows;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = idx / cols;
    int x = idx % cols;

    if(idx >= size) return;



    // bounds check
    int idx__xStart, idx__yStart, idx__xEnd, idx__yEnd;
    idx__xStart = (x-1 < 0) ? 0 : x-1;
    idx__xEnd = (x > cols - 1) ? cols - 1 : x;
    idx__yStart = (y-1 < 0) ? 0 : y-1;
    idx__yEnd = (y > rows - 1) ? rows - 1 : y;

    float *ptr__row1 = ((float*)((char*)inptr__currentFrame + idx__yStart * step));
    float *ptr__row2 = ((float*)((char*)inptr__currentFrame + y * step));
    float *ptr__row3 = ((float*)((char*)inptr__currentFrame + idx__yEnd * step));



    // calulate mean
    float mean = 0;

    mean +=      ptr__row1[idx__xStart]     +   (  4 *  ptr__row1[x])   +       ptr__row1[idx__xEnd];
    mean += (4 * ptr__row2[idx__xStart])    +   (-20 *  ptr__row2[x])   + (4 *  ptr__row2[idx__xEnd]);
    mean +=      ptr__row3[idx__xStart]     +   (  4 *  ptr__row3[x])   +       ptr__row3[idx__xEnd];
    mean /= 6;

    ((float*)((char*)outptr__meanMap + y * step))[x] = mean;



    // calulate clarity
    float clarity = 0;

    clarity += ptr__row1[idx__xStart]  + ptr__row1[x]   + ptr__row1[idx__xEnd];
    clarity += ptr__row2[idx__xStart]  + ptr__row2[x]   + ptr__row2[idx__xEnd];
    clarity += ptr__row3[idx__xStart]  + ptr__row3[x]   + ptr__row3[idx__xEnd];
    clarity /= 9;

    ((float*)((char*)outptr__clarityMap + y * step))[x] = clarity;

}

__host__ void DepthmapKernel::calculatedMeanAndClarityMap(cv::gpu::GpuMat &in__currentFrame, cv::gpu::GpuMat &out__meanMap, cv::gpu::GpuMat &out__clarityMap) {
    int cols = in__currentFrame.cols;
    int rows = in__currentFrame.rows;
    int frameSize = rows * cols;
    int threadSize = 1024;
    int blockSize = frameSize / threadSize;
    dv__calculatedMeanAndClarityMap<<<blockSize, threadSize>>>(in__currentFrame.ptr<float>(), in__currentFrame.step, cols, rows, out__meanMap.ptr<float>(), out__clarityMap.ptr<float>());
    cudaDeviceSynchronize();
}



//------------------------------------------------------------------
// CONTRAST
//------------------------------------------------------------------

__global__ void dv__calculateContrast(float *inptr__currentFrame, size_t step, int cols, int rows, float *inptr__meanMap, float *outptr__contrastMap) {

    int size = cols * rows;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = idx / cols;
    int x = idx % cols;

    if(idx > size) return;



    // bounds check
    int idx__xStart, idx__yStart, idx__xEnd, idx__yEnd;
    idx__xStart = (x-1 < 0) ? 0 : x-1;
    idx__xEnd = (x > cols - 1) ? cols - 1 : x;
    idx__yStart = (y-1 < 0) ? 0 : y-1;
    idx__yEnd = (y > rows - 1) ? rows - 1 : y;

    float *ptr__frameRow1 = ((float*)((char*)inptr__currentFrame + idx__yStart * step));
    float *ptr__frameRow2 = ((float*)((char*)inptr__currentFrame + y * step));
    float *ptr__frameRow3 = ((float*)((char*)inptr__currentFrame + idx__yEnd * step));

    float *ptr__meanMapRow1 = ((float*)((char*)inptr__meanMap + idx__yStart * step));
    float *ptr__meanMapRow2 = ((float*)((char*)inptr__meanMap + y * step));
    float *ptr__meanMapRow3 = ((float*)((char*)inptr__meanMap + idx__yEnd * step));



    // calulate contrast
    float contrast = 0;

    float val1 = ptr__frameRow1[idx__xStart]    - ptr__meanMapRow1[idx__xStart];
    float val2 = ptr__frameRow1[x]              - ptr__meanMapRow1[x];
    float val3 = ptr__frameRow1[idx__xEnd]      - ptr__meanMapRow1[idx__xEnd];
    float val4 = ptr__frameRow2[idx__xStart]    - ptr__meanMapRow2[idx__xStart];
    float val5 = ptr__frameRow2[x]              - ptr__meanMapRow2[x];
    float val6 = ptr__frameRow2[idx__xEnd]      - ptr__meanMapRow2[idx__xEnd];
    float val7 = ptr__frameRow3[idx__xStart]    - ptr__meanMapRow3[idx__xStart];
    float val8 = ptr__frameRow3[x]              - ptr__meanMapRow3[x];
    float val9 = ptr__frameRow3[idx__xEnd]      - ptr__meanMapRow3[idx__xEnd];

    contrast = (val1 * val1) + (val2 * val2) + (val3 * val3) + (val4 * val4) + (val5 * val5) + (val6 * val6) + (val7 * val7) + (val8 * val8) + (val9 * val9);
    contrast = sqrt(contrast / 9);

    ((float*)((char*)outptr__contrastMap + y * step))[x] = contrast;

}

__host__ void DepthmapKernel::calculateContrastMap(cv::gpu::GpuMat &in__currentFrame, cv::gpu::GpuMat &in__meanMap, cv::gpu::GpuMat &out__contrastMap) {
    int cols = in__currentFrame.cols;
    int rows = in__currentFrame.rows;
    int frameSize = rows * cols;
    int threadSize = 1024;
    int blockSize = frameSize / threadSize;
    dv__calculateContrast<<<blockSize, threadSize>>>(in__currentFrame.ptr<float>(), in__currentFrame.step, cols, rows, in__meanMap.ptr<float>(), out__contrastMap.ptr<float>());
    cudaDeviceSynchronize();
}



//------------------------------------------------------------------
// CLARITY AND CONTRAST FOR EVERY IMAGEBLOCK
//------------------------------------------------------------------

__global__ void dv__calculateClarityAndContrastPerImageblock(float* inptr__clarityMap, float* inptr__contrastMap, size_t step, int cols, int rows, int in__numberOfImageblocks, int in__imageBlockSize, float *outptr__clarities, float *outptr__contrasts) {

    // every thread is responsible for one imageblock
    int imageBlocksInX = cols / in__imageBlockSize;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = idx / imageBlocksInX;
    int x = idx % imageBlocksInX;
    int imageBlockStartY = y * in__imageBlockSize;
    int imageBlockStartX = x * in__imageBlockSize;

    if (idx >= in__numberOfImageblocks) return;

    // iterate over all pixels of the imageblock
    float clarity = 0;
    float contrast = 0;
    for (int ind_imageblockX = 0; ind_imageblockX < in__imageBlockSize; ind_imageblockX++) {
        for (int ind_imageblockY = 0; ind_imageblockY < in__imageBlockSize; ind_imageblockY++) {
            if ((imageBlockStartX + ind_imageblockX) >= cols || (imageBlockStartY + ind_imageblockY) >= rows) continue;
            clarity += ((float*)((char*)inptr__clarityMap + (imageBlockStartY + ind_imageblockY) * step))[imageBlockStartX + ind_imageblockX];
            contrast += ((float*)((char*)inptr__contrastMap + (imageBlockStartY + ind_imageblockY) * step))[imageBlockStartX + ind_imageblockX];
        }
    }

    outptr__clarities[idx] = clarity;
    outptr__contrasts[idx] = contrast;

}

__host__ void DepthmapKernel::calculateClarityAndContrastPerImageblock(cv::gpu::GpuMat &in__clarityMap, cv::gpu::GpuMat &in__contrastMap, int in__numberOfImageblocks, int in__imageBlockSize, float *outptr__clarities, float *outptr__contrasts) {
    int cols = in__clarityMap.cols;
    int rows = in__clarityMap.rows;
    int threadSize = 1024;
    int blockSize = in__numberOfImageblocks / threadSize;
    dv__calculateClarityAndContrastPerImageblock<<<blockSize, threadSize>>>(in__clarityMap.ptr<float>(), in__contrastMap.ptr<float>(), in__clarityMap.step, cols, rows, in__numberOfImageblocks, in__imageBlockSize, outptr__clarities, outptr__contrasts);
    cudaDeviceSynchronize();
}



//------------------------------------------------------------------
// FIND MAX CLARITY AND CONTRAST
//------------------------------------------------------------------

__global__ void dv__getMaxClarityAndContrast(float *inptr__clarities, float *inptr__contrasts, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfSize = (size / 2);
    bool isOdd = (2 * halfSize != size);

    // bound check
    if (idx < halfSize) {

        // add the values of two bins together
        int idx2 = idx + halfSize;
        if (isOdd && (idx == halfSize - 1)) {
            int idx3 = idx2 + 1;

            // clarity
            float clarity1 = inptr__clarities[idx];
            float clarity2 = inptr__clarities[idx2];
            float clarity3 = inptr__clarities[(idx3)];
            float maxClarity = (clarity1 < clarity2) ? clarity2 : clarity1;
            maxClarity = (maxClarity < clarity3) ? clarity3 : maxClarity;
            inptr__clarities[idx] = maxClarity;

            // contrast
            float contrast1 = inptr__clarities[idx];
            float contrast2 = inptr__clarities[idx2];
            float contrast3 = inptr__clarities[(idx3)];
            float maxContrast = (contrast1 < contrast2) ? contrast2 : contrast1;
            maxContrast = (maxContrast < contrast3) ? contrast3 : maxContrast;
            inptr__contrasts[idx] = maxContrast;
        }else {
            // clarity
            float clarity1 = inptr__clarities[idx];
            float clarity2 = inptr__clarities[idx2];
            float maxClarity = (clarity1 < clarity2) ? clarity2 : clarity1;
            inptr__clarities[idx] = maxClarity;

            // contrast
            float contrast1 = inptr__clarities[idx];
            float contrast2 = inptr__clarities[idx2];
            float maxContrast = (contrast1 < contrast2) ? contrast2 : contrast1;
            inptr__contrasts[idx] = maxContrast;
        }

    }
}

__host__ void DepthmapKernel::getMaxClarityAndContrast(float *inptr__clarities, float *inptr__contrasts, int in__numberOfImageblocks, float &maxClarity, float &maxContrast) {
    int threadSize = 1024;
    int blockSize = in__numberOfImageblocks / threadSize;

    int temp__size = in__numberOfImageblocks;

    printf("before doing stuff \n");

    float *temptr__clarities;
    float *temptr__contrasts;
    cudaMalloc((void**)&temptr__clarities, sizeof(float) * in__numberOfImageblocks);
    cudaMalloc((void**)&temptr__contrasts, sizeof(float) * in__numberOfImageblocks);
    cudaMemcpy(temptr__clarities, inptr__clarities, sizeof(float) * in__numberOfImageblocks, cudaMemcpyDeviceToDevice);
    cudaMemcpy(temptr__contrasts, inptr__contrasts, sizeof(float) * in__numberOfImageblocks, cudaMemcpyDeviceToDevice);
    for (int idx = 0; idx < in__numberOfImageblocks; idx++) {
        temptr__clarities[idx] = inptr__clarities[idx];
        temptr__contrasts[idx] = inptr__contrasts[idx];
    }

    printf("did stuff \n");

    while(temp__size > 1) {
        dv__getMaxClarityAndContrast<<<blockSize, threadSize>>>(temptr__clarities, temptr__contrasts, temp__size);
        cudaDeviceSynchronize();
        temp__size /= 2;
    }

    maxClarity = temptr__clarities[0];
    maxContrast = temptr__contrasts[0];
}

//------------------------------------------------------------------
// CALCULATE DEPTH FOR EVERY IMAGEBLOCK
//------------------------------------------------------------------

__global__ void dv__calculateDepthPerImageblock(float *inptr__clarities, float *inptr__contrasts, int size, float *outptr__depths) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    outptr__depths[idx] = (inptr__clarities[idx] + inptr__contrasts[idx]) / 2;
}

__host__ void DepthmapKernel::calculateDepthPerImageblock(float *inptr__clarities, float *inptr__contrasts, int size, float *outptr__depths) {
    int threadSize = 1024;
    int blockSize = size / threadSize;

    dv__calculateDepthPerImageblock<<<blockSize, threadSize>>>(inptr__clarities, inptr__contrasts, size, outptr__depths);
    cudaDeviceSynchronize();
}



//------------------------------------------------------------------
// CALCULATE MEAN CB AN CR VALUES
//------------------------------------------------------------------

__global__ void dv__calculateMeanCbAndCrValues(float* inptr__cbMap, float* inptr__crMap, size_t step, int cols, int rows, int in_numberOfImageblocks, int in__imageBlockSize, float *outptr__meanCb, float *outptr__meanCr) {

    // every thread is responsible for one imageblock
    int imageBlocksInX = cols / in__imageBlockSize;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = idx / imageBlocksInX;
    int x = idx % imageBlocksInX;
    int imageBlockStartY = y * in__imageBlockSize;
    int imageBlockStartX = x * in__imageBlockSize;

    if (idx >= in_numberOfImageblocks) return;

    // iterate over all pixels of the imageblock
    float cb = 0;
    float cr = 0;
    int count = 0;
    for (int ind_imageblockX = 0; ind_imageblockX < in__imageBlockSize; ind_imageblockX++) {
        for (int ind_imageblockY = 0; ind_imageblockY < in__imageBlockSize; ind_imageblockY++) {
            if ((imageBlockStartX + ind_imageblockX) >= cols || (imageBlockStartY + ind_imageblockY) >= rows) continue;
            cb += ((float*)((char*)inptr__cbMap + (imageBlockStartY + ind_imageblockY) * step))[imageBlockStartX + ind_imageblockX];
            cr += ((float*)((char*)inptr__crMap + (imageBlockStartY + ind_imageblockY) * step))[imageBlockStartX + ind_imageblockX];
            count++;
        }
    }
    cb /= count;
    cr /= count;

    // set results
    outptr__meanCb[idx] = cb;
    outptr__meanCr[idx] = cr;

}

__host__ void DepthmapKernel::calculateMeanCbAndCrValues(cv::gpu::GpuMat &in__cbMap, cv::gpu::GpuMat &in__crMap, int in__numberOfImageblocks, int in__imageBlockSize, float *outptr__meanCb, float *outptr__meanCr) {
    int cols = in__cbMap.cols;
    int rows = in__cbMap.rows;
    int threadSize = 1024;
    int blockSize = in__numberOfImageblocks / threadSize;
    dv__calculateMeanCbAndCrValues<<<blockSize, threadSize>>>(in__cbMap.ptr<float>(), in__crMap.ptr<float>(), in__cbMap.step, cols, rows, in__numberOfImageblocks, in__imageBlockSize, outptr__meanCb, outptr__meanCr);
    cudaDeviceSynchronize();
}



//------------------------------------------------------------------
// MERGE ADJACENT IMAGEBLOCKS WITH SIMILAR MEAN CB AND CR VALUES
//------------------------------------------------------------------

__global__ void dv__mergeNeighborImageblocks(float* inptr__meanCb, float* inptr__meanCr, int numberOfImageBlocks, int imageBlockCols, float threshold, float *outptr__neighbors) {

    // every thread is responsible for one imageblock
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numberOfImageBlocks) return;

    float cb = inptr__meanCb[idx];
    float cr = inptr__meanCr[idx];



    // check wether imageblock is similar to his upper or left neighbor
    int y = idx / imageBlockCols;
    int x = idx % imageBlockCols;
    int upper = y - 1;
    int left = x - 1;

    if (upper > 0) {
        int idx__upperNeighbor = upper * imageBlockCols + x;
        float cbUpperNeighbor = inptr__meanCb[idx__upperNeighbor];
        float crUpperNeighbor = inptr__meanCr[idx__upperNeighbor];
        float deltaCb = (cb - cbUpperNeighbor) * (cb - cbUpperNeighbor);
        float deltaCr = (cr - crUpperNeighbor) * (cr - crUpperNeighbor);

        if (deltaCb + deltaCr < threshold) {    // sum of squared differences
            outptr__neighbors[idx] = idx__upperNeighbor;
            return;
        }

    }

    if (left > 0) {
        int idx__leftNeighbor = y * imageBlockCols + left;
        float cbLeftNeighbor = inptr__meanCb[idx__leftNeighbor];
        float crLeftNeighbor = inptr__meanCr[idx__leftNeighbor];
        float deltaCb = (cb - cbLeftNeighbor) * (cb - cbLeftNeighbor);
        float deltaCr = (cr - crLeftNeighbor) * (cr - crLeftNeighbor);

        if (deltaCb + deltaCr < threshold) {    // sum of squared differences
            outptr__neighbors[idx] = idx__leftNeighbor;
            return;
        }
    }

    // imageblock has no neighbors
    outptr__neighbors[idx] = idx;

}

__host__ void DepthmapKernel::mergeNeighborImageblocks(float* inptr__meanCb, float* inptr__meanCr, int in__numberOfImageblocks, int imageBlockCols, float threshold, float *outptr__neighbors) {
    int threadSize = 1024;
    int blockSize = in__numberOfImageblocks / threadSize;
    dv__mergeNeighborImageblocks<<<blockSize, threadSize>>>(inptr__meanCb, inptr__meanCr, in__numberOfImageblocks, imageBlockCols, threshold, outptr__neighbors);
    cudaDeviceSynchronize();
}



//------------------------------------------------------------------
// ADJUST DISTANCE AND DEPTHLEVEL
//------------------------------------------------------------------

__global__ void dv__adjustDepthAndDistancelevel(float* inptr__neighbors, float *inptr__depths, int numberOfImageBlocks, float *outptr__adjustedDepths, int *outptr__distancelevels) {

    // every thread is responsible for one imageblock
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numberOfImageBlocks) return;

    int count = 0;
    float depth = 0;
    float maxdepth = 0;
    int distancelevel = 0;
    int *maskedImageblocks = new int[numberOfImageBlocks];

    // iterate over all imageblocks and calculate mean depth
    for (int idx__neighbor =0; idx__neighbor < numberOfImageBlocks; idx__neighbor++) {
        if(inptr__neighbors[idx__neighbor] == idx) {
            float temp__depth = inptr__depths[idx__neighbor];
            if (temp__depth > maxdepth) maxdepth = temp__depth;
            depth += temp__depth;
            maskedImageblocks[idx__neighbor] = 1;
            count++;
        }else {
            maskedImageblocks[idx__neighbor] = 0;
        }
    }

    if (count == 0) return; // this imageblock is part of another imageblock so therefore it can be ignored

    depth /= count;

    // estimate distance level
    if (depth < maxdepth / 3) {                 // FAR
        distancelevel = 0;
    }else if (depth <= (2 * maxdepth) / 3) {    // MIDDLE
        distancelevel = 1;
    }else {                                     // NEAR
        distancelevel = 2;
    }

    // set adjusted depth and distance level for all imageblocks that belong to the same object
    for (int idx__neighbor =0; idx__neighbor < numberOfImageBlocks; idx__neighbor++) {
        if (maskedImageblocks[idx__neighbor] == 1) {
            outptr__adjustedDepths[idx__neighbor] = depth;
            outptr__distancelevels[idx__neighbor] = distancelevel;
        }
    }

}

__host__ void DepthmapKernel::adjustDepthAndDistancelevel(float* inptr__neighbors, float *inptr__depths, int numberOfImageBlocks, float *outptr__adjustedDepths, int *outptr__distancelevels) {
    int threadSize = 1024;
    int blockSize = numberOfImageBlocks / threadSize;
    dv__adjustDepthAndDistancelevel<<<blockSize, threadSize>>>(inptr__neighbors, inptr__depths, numberOfImageBlocks, outptr__adjustedDepths, outptr__distancelevels);
    cudaDeviceSynchronize();
}



