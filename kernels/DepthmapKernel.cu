#include "DepthmapKernel.h"

//------------------------------------------------------------------
// MEAN AND CLARITY
//------------------------------------------------------------------

__global__ void dv__calculatedMeanAndClarityMap(float *inptr__currentFrame, size_t step, int cols, int rows, float *outptr__meanMap, float *outptr__clarityMap) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int size = cols * rows;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = idx / cols;
    int x = idx % cols;


    //---------------------------------------------------------------------------------------------
    //bounds check
    //---------------------------------------------------------------------------------------------
    if(idx >= size) return;


    //---------------------------------------------------------------------------------------------
    // edge treatment the 3x3 mask
    //---------------------------------------------------------------------------------------------
    int idx__xStart, idx__yStart, idx__xEnd, idx__yEnd;
    idx__xStart = (x-1 < 0) ? 0 : x-1;
    idx__xEnd = (x+1 > cols - 1) ? cols - 1 : x+1;
    idx__yStart = (y-1 < 0) ? 0 : y-1;
    idx__yEnd = (y+1 > rows - 1) ? rows - 1 : y+1;


    //---------------------------------------------------------------------------------------------
    // get rows of the 3x3 mask
    //---------------------------------------------------------------------------------------------
    float *ptr__row1 = ((float*)((char*)inptr__currentFrame + idx__yStart * step));
    float *ptr__row2 = ((float*)((char*)inptr__currentFrame + y * step));
    float *ptr__row3 = ((float*)((char*)inptr__currentFrame + idx__yEnd * step));


    //---------------------------------------------------------------------------------------------
    // calulate clarity
    //---------------------------------------------------------------------------------------------
    float clarity = 0;
    clarity +=      ptr__row1[idx__xStart]     +   (  4 *  ptr__row1[x])   +       ptr__row1[idx__xEnd];
    clarity += (4 * ptr__row2[idx__xStart])    +   (-20 *  ptr__row2[x])   + (4 *  ptr__row2[idx__xEnd]);
    clarity +=      ptr__row3[idx__xStart]     +   (  4 *  ptr__row3[x])   +       ptr__row3[idx__xEnd];
    clarity /= 6;
    ((float*)((char*)outptr__clarityMap + y * step))[x] = clarity;


    //---------------------------------------------------------------------------------------------
    // calulate mean
    //---------------------------------------------------------------------------------------------
    float mean = 0;
    mean += ptr__row1[idx__xStart]  + ptr__row1[x]   + ptr__row1[idx__xEnd];
    mean += ptr__row2[idx__xStart]  + ptr__row2[x]   + ptr__row2[idx__xEnd];
    mean += ptr__row3[idx__xStart]  + ptr__row3[x]   + ptr__row3[idx__xEnd];
    mean /= 9;
    ((float*)((char*)outptr__meanMap + y * step))[x] = mean;

}

__host__ void DepthmapKernel::calculatedMeanAndClarityMap(cv::gpu::GpuMat &in__currentFrame, cv::gpu::GpuMat &out__meanMap, cv::gpu::GpuMat &out__clarityMap) {
    int cols = in__currentFrame.cols;
    int rows = in__currentFrame.rows;
    int frameSize = rows * cols;
    int threadSize = 1024;
    int blockSize = (frameSize / threadSize) + 1;
    if (blockSize <= 0) blockSize = 1;
    dv__calculatedMeanAndClarityMap<<<blockSize, threadSize>>>(in__currentFrame.ptr<float>(), in__currentFrame.step, cols, rows, out__meanMap.ptr<float>(), out__clarityMap.ptr<float>());
    cudaDeviceSynchronize();
}



//------------------------------------------------------------------
// CONTRAST
//------------------------------------------------------------------

__global__ void dv__calculateContrast(float *inptr__currentFrame, size_t step, int cols, int rows, float *inptr__meanMap, float *outptr__contrastMap) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int size = cols * rows;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = idx / cols;
    int x = idx % cols;


    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if(idx >= size) return;


    //---------------------------------------------------------------------------------------------
    // edge treatment for the 3x3 mask
    //---------------------------------------------------------------------------------------------
    int idx__xStart, idx__yStart, idx__xEnd, idx__yEnd;
    idx__xStart = (x-1 < 0) ? 0 : x-1;
    idx__xEnd = (x+1 > cols - 1) ? cols - 1 : x+1;
    idx__yStart = (y-1 < 0) ? 0 : y-1;
    idx__yEnd = (y+1 > rows - 1) ? rows - 1 : y+1;


    //---------------------------------------------------------------------------------------------
    // rows of imagesensitivities
    //---------------------------------------------------------------------------------------------
    float *ptr__frameRow1 = ((float*)((char*)inptr__currentFrame + idx__yStart * step));
    float *ptr__frameRow2 = ((float*)((char*)inptr__currentFrame + y * step));
    float *ptr__frameRow3 = ((float*)((char*)inptr__currentFrame + idx__yEnd * step));


    //---------------------------------------------------------------------------------------------
    // rows of the means
    //---------------------------------------------------------------------------------------------
    float *ptr__meanMapRow1 = ((float*)((char*)inptr__meanMap + idx__yStart * step));
    float *ptr__meanMapRow2 = ((float*)((char*)inptr__meanMap + y * step));
    float *ptr__meanMapRow3 = ((float*)((char*)inptr__meanMap + idx__yEnd * step));



    //---------------------------------------------------------------------------------------------
    // calculate contrast
    //---------------------------------------------------------------------------------------------
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
    int blockSize = (frameSize / threadSize) + 1;
    if (blockSize <= 0) blockSize = 1;
    dv__calculateContrast<<<blockSize, threadSize>>>(in__currentFrame.ptr<float>(), in__currentFrame.step, cols, rows, in__meanMap.ptr<float>(), out__contrastMap.ptr<float>());
    cudaDeviceSynchronize();
}



//------------------------------------------------------------------
// CLARITY AND CONTRAST FOR EVERY IMAGEBLOCK
//------------------------------------------------------------------

__global__ void dv__calculateClarityAndContrastPerImageblock(float* inptr__clarityMap, float* inptr__contrastMap, size_t step, int cols, int rows, int in__numberOfImageblocks, int in__imageBlockSize, float *outptr__clarities, float *outptr__contrasts) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int imageBlocksInX = cols / in__imageBlockSize;
    int imageBlocksInY = rows / in__imageBlockSize;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = idx / imageBlocksInX;
    int x = idx % imageBlocksInX;
    int imageBlockStartY = y * in__imageBlockSize;
    int imageBlockStartX = x * in__imageBlockSize;


    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= in__numberOfImageblocks || x >= imageBlocksInX || y >= imageBlocksInY ) return;


    //---------------------------------------------------------------------------------------------
    // sum up all clarities and all contrasts of all pixels within the imageblock
    //---------------------------------------------------------------------------------------------
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
    int blockSize = (in__numberOfImageblocks / threadSize) + 1;
    if (blockSize <= 0) blockSize = 1;
    dv__calculateClarityAndContrastPerImageblock<<<blockSize, threadSize>>>(in__clarityMap.ptr<float>(), in__contrastMap.ptr<float>(), in__clarityMap.step, cols, rows, in__numberOfImageblocks, in__imageBlockSize, outptr__clarities, outptr__contrasts);
    cudaDeviceSynchronize();
}



//------------------------------------------------------------------
// FIND MAX CLARITY AND CONTRAST
//------------------------------------------------------------------

__global__ void dv__getMaxClarityAndContrast(float *inptr__clarities, float *inptr__contrasts, int size) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfSize = (size / 2);
    bool isOdd = (2 * halfSize != size);


    //---------------------------------------------------------------------------------------------
    // check bounds
    //---------------------------------------------------------------------------------------------
    if (idx < halfSize) {


        //---------------------------------------------------------------------------------------------
        // divide array of clarities(contrasts) in a left and a right part
        // compare a values of the left side with one of the right side
        // store the bigger values at the position of the left value
        //---------------------------------------------------------------------------------------------
        int idx2 = idx + halfSize;
        // in case the array is odd and we are at the end of the left part
        // then compare this value with the last two values of the right part
        if (isOdd && (idx == halfSize - 1)) {
            int idx3 = idx2 + 1;
            // calculate clarity
            float clarity1 = inptr__clarities[idx];
            float clarity2 = inptr__clarities[idx2];
            float clarity3 = inptr__clarities[(idx3)];
            float maxClarity = (clarity1 < clarity2) ? clarity2 : clarity1;
            maxClarity = (maxClarity < clarity3) ? clarity3 : maxClarity;
            inptr__clarities[idx] = maxClarity;
            // calculate contrast
            float contrast1 = inptr__clarities[idx];
            float contrast2 = inptr__clarities[idx2];
            float contrast3 = inptr__clarities[(idx3)];
            float maxContrast = (contrast1 < contrast2) ? contrast2 : contrast1;
            maxContrast = (maxContrast < contrast3) ? contrast3 : maxContrast;
            inptr__contrasts[idx] = maxContrast;
        }else {
            // calulate clarity
            float clarity1 = inptr__clarities[idx];
            float clarity2 = inptr__clarities[idx2];
            float maxClarity = (clarity1 < clarity2) ? clarity2 : clarity1;
            inptr__clarities[idx] = maxClarity;
            // calculate contrast
            float contrast1 = inptr__clarities[idx];
            float contrast2 = inptr__clarities[idx2];
            float maxContrast = (contrast1 < contrast2) ? contrast2 : contrast1;
            inptr__contrasts[idx] = maxContrast;
        }

    }
}

__host__ void DepthmapKernel::getMaxClarityAndContrast(float *inptr__clarities, float *inptr__contrasts, int in__numberOfImageblocks, float &maxClarity, float &maxContrast) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int threadSize = 1024;
    int blockSize = (in__numberOfImageblocks / threadSize) + 1;
    if (blockSize <= 0) blockSize = 1;
    int temp__size = in__numberOfImageblocks;
    printf("before doing stuff \n");


    //---------------------------------------------------------------------------------------------
    // copy values of clarity and contrast arrays to temporary arrays that can be altered
    //---------------------------------------------------------------------------------------------
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


    //---------------------------------------------------------------------------------------------
    // collect all values by halve the temporary arrays and saveing the bigger values in the left half
    // until the sive of the temporary array is 1 and the biggest value is found
    //---------------------------------------------------------------------------------------------
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

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= size) return;


    //---------------------------------------------------------------------------------------------
    // get depth by even clarity and contrast
    //---------------------------------------------------------------------------------------------
    outptr__depths[idx] = inptr__clarities[idx] + 2* inptr__contrasts[idx];

}

__host__ void DepthmapKernel::calculateDepthPerImageblock(float *inptr__clarities, float *inptr__contrasts, int size, float *outptr__depths) {
    int threadSize = 1024;
    int blockSize = (size / threadSize) + 1;
    if (blockSize <= 0) blockSize = 1;
    dv__calculateDepthPerImageblock<<<blockSize, threadSize>>>(inptr__clarities, inptr__contrasts, size, outptr__depths);
    cudaDeviceSynchronize();
}



//------------------------------------------------------------------
// CALCULATE MEAN CB AN CR VALUES
//------------------------------------------------------------------

__global__ void dv__calculateMeanCbAndCrValues(float* inptr__cbMap, float* inptr__crMap, size_t step, int cols, int rows, int in_numberOfImageblocks, int in__imageBlockSize, float *outptr__meanCb, float *outptr__meanCr) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int imageBlocksInX = cols / in__imageBlockSize;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = idx / imageBlocksInX;
    int x = idx % imageBlocksInX;
    int imageBlockStartY = y * in__imageBlockSize;
    int imageBlockStartX = x * in__imageBlockSize;


    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= in_numberOfImageblocks) return;


    //---------------------------------------------------------------------------------------------
    // calculate mean Cb and Cr by summing up all values of the pixels within the imageblock
    //---------------------------------------------------------------------------------------------
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
    outptr__meanCb[idx] = cb;
    outptr__meanCr[idx] = cr;

}

__host__ void DepthmapKernel::calculateMeanCbAndCrValues(cv::gpu::GpuMat &in__cbMap, cv::gpu::GpuMat &in__crMap, int in__numberOfImageblocks, int in__imageBlockSize, float *outptr__meanCb, float *outptr__meanCr) {
    int cols = in__cbMap.cols;
    int rows = in__cbMap.rows;
    int threadSize = 1024;
    int blockSize = (in__numberOfImageblocks / threadSize) + 1;
    if (blockSize <= 0) blockSize = 1;
    dv__calculateMeanCbAndCrValues<<<blockSize, threadSize>>>(in__cbMap.ptr<float>(), in__crMap.ptr<float>(), in__cbMap.step, cols, rows, in__numberOfImageblocks, in__imageBlockSize, outptr__meanCb, outptr__meanCr);
    cudaDeviceSynchronize();
}



//------------------------------------------------------------------
// MERGE ADJACENT IMAGEBLOCKS WITH SIMILAR MEAN CB AND CR VALUES
//------------------------------------------------------------------

__host__ void DepthmapKernel::dv__mergeNeighborImageblocks(float* inptr__meanCb, float* inptr__meanCr, int numberOfImageBlocks, int numberOfImageblocksInX, float threshold, float *outptr__neighbors) {

    for (int idx = 0; idx < numberOfImageBlocks; idx++) {
        //---------------------------------------------------------------------------------------------
        // setup variables
        //---------------------------------------------------------------------------------------------
        int y = idx / numberOfImageblocksInX;
        int x = idx % numberOfImageblocksInX;
        int upper = y - 1;
        int left = x - 1;



        //---------------------------------------------------------------------------------------------
        // bounds check
        //---------------------------------------------------------------------------------------------
        if (idx >= numberOfImageBlocks) return;


        //---------------------------------------------------------------------------------------------
        // get mean Cb and Cr value of this threads imageblock
        //---------------------------------------------------------------------------------------------
        float cb = inptr__meanCb[idx];
        float cr = inptr__meanCr[idx];


        //---------------------------------------------------------------------------------------------
        // check wether imageblock is similar to his upper or left neighbor
        // in case imageblock is similar to a neighbor then set this neighbors id in the neighbors array
        // at the position of the current imageblocks id:
        // neighbors[current_imageblocks_id] = neighbors_id
        //---------------------------------------------------------------------------------------------
        if (upper >= 0) {
            int idx__upperNeighbor = (upper * numberOfImageblocksInX) + x;
            float cbUpperNeighbor = inptr__meanCb[idx__upperNeighbor];
            float crUpperNeighbor = inptr__meanCr[idx__upperNeighbor];
            float deltaCb = (cb - cbUpperNeighbor) * (cb - cbUpperNeighbor);
            float deltaCr = (cr - crUpperNeighbor) * (cr - crUpperNeighbor);

            if (deltaCb + deltaCr < threshold) {    // sum of squared differences
                outptr__neighbors[idx] = outptr__neighbors[idx__upperNeighbor];
                continue;
            }

        }
        if (left >= 0) {
            int idx__leftNeighbor = (y * numberOfImageblocksInX) + left;
            float cbLeftNeighbor = inptr__meanCb[idx__leftNeighbor];
            float crLeftNeighbor = inptr__meanCr[idx__leftNeighbor];
            float deltaCb = (cb - cbLeftNeighbor) * (cb - cbLeftNeighbor);
            float deltaCr = (cr - crLeftNeighbor) * (cr - crLeftNeighbor);

            if (deltaCb + deltaCr < threshold) {    // sum of squared differences
                outptr__neighbors[idx] = outptr__neighbors[idx__leftNeighbor];
                continue;
            }
        }


        //---------------------------------------------------------------------------------------------
        // imageblock has no neighbors that are similar in Cb and Cr value
        //---------------------------------------------------------------------------------------------
        outptr__neighbors[idx] = idx;
    }
}




__host__ void DepthmapKernel::mergeNeighborImageblocks(float* inptr__meanCb, float* inptr__meanCr, int in__numberOfImageblocks, int numberOfImageblocksInX, float threshold, float *outptr__neighbors) {
    dv__mergeNeighborImageblocks(inptr__meanCb, inptr__meanCr, in__numberOfImageblocks, numberOfImageblocksInX, threshold, outptr__neighbors);
}



//------------------------------------------------------------------
// ADJUST DISTANCE AND DEPTHLEVEL
//------------------------------------------------------------------

__global__ void dv__adjustDepthAndDistancelevel(float* inptr__neighbors, float *inptr__depths, int numberOfImageBlocks, float *outptr__adjustedDepths, int *outptr__distancelevels) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = 0;
    float depth = 0;
    float maxdepth = 0;
    int distancelevel = 0;
    int *maskedImageblocks = new int[numberOfImageBlocks];


    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= numberOfImageBlocks) {
        delete[] maskedImageblocks;
        return;
    }


    //---------------------------------------------------------------------------------------------
    // for all merged imageblocks sum up all depths
    //---------------------------------------------------------------------------------------------
    for (int idx__neighbor = 0; idx__neighbor < numberOfImageBlocks; idx__neighbor++) {
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


    //---------------------------------------------------------------------------------------------
    // calculate mean depth
    //---------------------------------------------------------------------------------------------
    if (count > 1) {
        depth /= count;
    }


    //---------------------------------------------------------------------------------------------
    // calculate distance level for all merged imageblocks
    //---------------------------------------------------------------------------------------------
    if (depth < maxdepth / 3) {                 // FAR
        distancelevel = 0;
    }else if (depth <= (2 * maxdepth) / 3) {    // MIDDLE
        distancelevel = 1;
    }else {                                     // NEAR
        distancelevel = 2;
    }


    //---------------------------------------------------------------------------------------------
    // set adjusted depth and distance level for all imageblocks that belong to the same object
    //---------------------------------------------------------------------------------------------
    for (int idx__neighbor = 0; idx__neighbor < numberOfImageBlocks; idx__neighbor++) {
        if (maskedImageblocks[idx__neighbor] == 1) {
            outptr__adjustedDepths[idx__neighbor] = depth;
            outptr__distancelevels[idx__neighbor] = distancelevel;
        }
    }


    //---------------------------------------------------------------------------------------------
    // release memory
    //---------------------------------------------------------------------------------------------
    delete[] maskedImageblocks;

}

__host__ void DepthmapKernel::adjustDepthAndDistancelevel(float* inptr__neighbors, float *inptr__depths, int numberOfImageBlocks, float *outptr__adjustedDepths, int *outptr__distancelevels) {
    int threadSize = 1024;
    int blockSize = (numberOfImageBlocks / threadSize) + 1;
    if (blockSize <= 0) blockSize = 1;
    dv__adjustDepthAndDistancelevel<<<blockSize, threadSize>>>(inptr__neighbors, inptr__depths, numberOfImageBlocks, outptr__adjustedDepths, outptr__distancelevels);
    cudaDeviceSynchronize();
}



//------------------------------------------------------------------
// FIND MAX DEPTH
//------------------------------------------------------------------

__global__ void dv__getMaxDepth(float *inptr__depths, int size) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfSize = (size / 2);
    bool isOdd = (2 * halfSize != size);


    //---------------------------------------------------------------------------------------------
    // check bounds
    //---------------------------------------------------------------------------------------------
    if (idx < halfSize) {


        //---------------------------------------------------------------------------------------------
        // divide array of depths in a left and a right part
        // compare a values of the left side with one of the right side
        // store the bigger values at the position of the left value
        //---------------------------------------------------------------------------------------------
        int idx2 = idx + halfSize;
        // in case the array is odd and we are at the end of the left part
        // then compare this value with the last two values of the right part
        if (isOdd && (idx == halfSize - 1)) {
            int idx3 = idx2 + 1;
            float depth1 = inptr__depths[idx];
            float depth2 = inptr__depths[idx2];
            float depth3 = inptr__depths[(idx3)];
            float maxDepth = (depth1 < depth2) ? depth2 : depth1;
            maxDepth = (maxDepth < depth3) ? depth3 : maxDepth;
            inptr__depths[idx] = maxDepth;
        }else {
            float depth1 = inptr__depths[idx];
            float depth2 = inptr__depths[idx2];
            float maxDepth = (depth1 < depth2) ? depth2 : depth1;
            inptr__depths[idx] = maxDepth;
        }

    }
}

__host__ void DepthmapKernel::getMaxDepth(float *inptr__depths, int in__numberOfImageblocks, float *outptr__maxDepth) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int threadSize = 1024;
    int blockSize = (in__numberOfImageblocks / threadSize) + 1;
    if (blockSize <= 0) blockSize = 1;
    int temp__size = in__numberOfImageblocks;


    //---------------------------------------------------------------------------------------------
    // copy values of clarity and contrast arrays to temporary arrays that can be altered
    //---------------------------------------------------------------------------------------------
    float *temptr__depths;
    cudaMalloc((void**)&temptr__depths, sizeof(float) * in__numberOfImageblocks);
    cudaMemcpy(temptr__depths, inptr__depths, sizeof(float) * in__numberOfImageblocks, cudaMemcpyDeviceToDevice);


    //---------------------------------------------------------------------------------------------
    // collect all values by halve the temporary arrays and saveing the bigger values in the left half
    // until the sive of the temporary array is 1 and the biggest value is found
    //---------------------------------------------------------------------------------------------
    while(temp__size > 1) {
        dv__getMaxDepth<<<blockSize, threadSize>>>(temptr__depths, temp__size);
        cudaDeviceSynchronize();
        temp__size /= 2;
    }


    //---------------------------------------------------------------------------------------------
    // copy back results
    //---------------------------------------------------------------------------------------------
    cudaMemcpy(outptr__maxDepth, temptr__depths, sizeof(float), cudaMemcpyDeviceToDevice);


    //---------------------------------------------------------------------------------------------
    // release memory
    //---------------------------------------------------------------------------------------------
    cudaFree(temptr__depths);

}



//------------------------------------------------------------------
// NORMALIZE DEPTH
//------------------------------------------------------------------

__global__ void dv__normalizeDepth(float *inptr__adjustedDepth, float *inptr__maxDepth, int in__numberOfImageblocks) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    //---------------------------------------------------------------------------------------------
    //bounds check
    //---------------------------------------------------------------------------------------------
    if(idx >= in__numberOfImageblocks) return;


    //---------------------------------------------------------------------------------------------
    // calculate normalized depth
    //---------------------------------------------------------------------------------------------
    float adjustedDepth = inptr__adjustedDepth[idx];
    inptr__adjustedDepth[idx] = adjustedDepth / *inptr__maxDepth;

}

__host__ void DepthmapKernel::normalizeDepth(float *inptr__adjustedDepth, float *inptr__maxDepth, int in__numberOfImageblocks) {
    int threadSize = 1024;
    int blockSize = (in__numberOfImageblocks / threadSize) + 1;
    if (blockSize <= 0) blockSize = 1;
    dv__normalizeDepth<<<blockSize, threadSize>>>(inptr__adjustedDepth, inptr__maxDepth, in__numberOfImageblocks);
    cudaDeviceSynchronize();
}



//------------------------------------------------------------------
// FILL DEPTHMAP WITH IMAGEBLOCK DEPTH
//------------------------------------------------------------------

__global__ void dv__fillDepthMap(float* inptr__depthMap, size_t step, int cols, int rows, int in_numberOfImageblocks, int in__imageBlockSize, float *inptr__depth) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int numberOfImageblocksInX = cols / in__imageBlockSize;
    if (numberOfImageblocksInX <= 0) numberOfImageblocksInX = 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = idx / numberOfImageblocksInX;
    int x = idx % numberOfImageblocksInX;
    int imageBlockStartY = y * in__imageBlockSize;
    int imageBlockStartX = x * in__imageBlockSize;


    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= in_numberOfImageblocks) return;


    //---------------------------------------------------------------------------------------------
    // get mean depth of the imageblock
    //---------------------------------------------------------------------------------------------
    float depth = 0;
    depth = inptr__depth[idx];


    //---------------------------------------------------------------------------------------------
    // fill all pixels in the depthmap within the imageblock with the imageblocks depth
    //---------------------------------------------------------------------------------------------
    for (int ind_imageblockX = 0; ind_imageblockX < in__imageBlockSize; ind_imageblockX++) {
        for (int ind_imageblockY = 0; ind_imageblockY < in__imageBlockSize; ind_imageblockY++) {
            if ((imageBlockStartX + ind_imageblockX) >= cols || (imageBlockStartY + ind_imageblockY) >= rows) continue;
            ((float*)((char*)inptr__depthMap + (imageBlockStartY + ind_imageblockY) * step))[imageBlockStartX + ind_imageblockX] = depth;
        }
    }

}

__host__ void DepthmapKernel::fillDepthMap(cv::gpu::GpuMat &in__depthMap, int in__numberOfImageblocks, int in__imageBlockSize, float *inptr__depth) {
    int threadSize = 1024;
    int blockSize = (in__numberOfImageblocks / threadSize) + 1;
    if (blockSize <= 0) blockSize = 1;
    dv__fillDepthMap<<<blockSize, threadSize>>>(in__depthMap.ptr<float>(), in__depthMap.step , in__depthMap.cols, in__depthMap.rows, in__numberOfImageblocks, in__imageBlockSize, inptr__depth);
    cudaDeviceSynchronize();
}
