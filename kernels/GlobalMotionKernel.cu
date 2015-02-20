#include "GlobalMotionKernel.h"

#define imin(a,b) (a<b?a:b)

//---------------------------------------------------------------------------------------------
// subsample optical flow to create a smaller coarse flowfield
//---------------------------------------------------------------------------------------------

__global__ void dv__createCoarse3DFlow(float *dinptr__3DFlowX, float *dinptr__3DFlowY, size_t in__stepFlow, int in__rowsFlow, int in__colsFlow, int in__coarseLevel, size_t in__stepCoarse, float *outptr__coarse3DFlowX, float *outptr__coarse3DFlowY) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blocksInX = in__colsFlow / in__coarseLevel;
    int blocksInY = in__rowsFlow / in__coarseLevel;
    int y = idx / blocksInX;
    int x = idx % blocksInX;
    int blockX = x * in__coarseLevel;
    int blockY = y * in__coarseLevel;


    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= blocksInX * blocksInY) return;


    //---------------------------------------------------------------------------------------------
    // subsample flow
    //---------------------------------------------------------------------------------------------
    float meanX = 0.0f;
    float meanY = 0.0f;
    int counter = 0;
    for (int j = 0; j < in__coarseLevel; j++) {
        for (int i = 0; i < in__coarseLevel; i++) {
            if (blockX + i >= in__colsFlow || blockY + j >= in__rowsFlow) continue;
            meanX += ((float*)((char*)dinptr__3DFlowX + (blockY + j) * in__stepFlow))[blockX + i];
            meanY += ((float*)((char*)dinptr__3DFlowY + (blockY + j) * in__stepFlow))[blockX + i];
            counter++;
        }
    }
    if (counter != 0) {
        meanX /= counter;
        meanY /= counter;
    }


    //---------------------------------------------------------------------------------------------
    // save result in coarse flow field
    //---------------------------------------------------------------------------------------------
    ((float*)((char*)outptr__coarse3DFlowX + y * in__stepCoarse))[x] = meanX;
    ((float*)((char*)outptr__coarse3DFlowY + y * in__stepCoarse))[x] = meanY;

}

void GlobalMotionKernel::createCoarse3DFlow(cv::gpu::GpuMat &din__3DFlowX, cv::gpu::GpuMat &din__3DFlowY, int in__coarseLevel, cv::gpu::GpuMat &dout__coarse3DFlowX, cv::gpu::GpuMat &dout__coarse3DFlowY) {

    int cols = din__3DFlowX.cols;
    int rows = din__3DFlowX.rows;
    int frameSize = rows * cols;
    int threadSize = 1024;
    int blockSize = (frameSize / threadSize) + 1;
    if (blockSize <= 0) blockSize = 1;

    dv__createCoarse3DFlow<<<blockSize, threadSize>>>(din__3DFlowX.ptr<float>(), din__3DFlowY.ptr<float>(), din__3DFlowX.step, rows, cols, in__coarseLevel, dout__coarse3DFlowX.step, dout__coarse3DFlowX.ptr<float>(), dout__coarse3DFlowY.ptr<float>());
}




//---------------------------------------------------------------------------------------------
// iterate over the whole image and calculate SSD for every pixel
//---------------------------------------------------------------------------------------------

__global__ void dv__iterateOverTheWholeImage(float *dinptr__3DFlowX, float *dinptr__3DFlowY, size_t in__stepFlow, int in__rowsFlow, int in__colsFlow, int startX, int startY, int endX, int endY, int w, float in__threshold, size_t in__stepSSDs, float *doutptr__SSDs) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sizeX = (endX - startX) + 1;
    int sizeY = (endY - startY) + 1;
    int size = sizeX * sizeY;
    int posWithinFieldY = idx / sizeX;
    int posWithinFieldX = idx % sizeX;
    int centerY = startY + posWithinFieldY;
    int centerX = startX + posWithinFieldX;


    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= size) return;


    float weights = 0.0f;
    float ssds = 0.0f;
    //---------------------------------------------------------------------------------------------
    // iterate over filter
    //---------------------------------------------------------------------------------------------
    for (int v = -w; v <= w; v++) {
        for (int u = -w; u <= w; u++) {

            int ind__flowX = centerX + u;
            int ind__flowY = centerY + v;


            //---------------------------------------------------------------------------------------------
            // bounds check
            //---------------------------------------------------------------------------------------------
            if (ind__flowX < 0 || ind__flowY < 0 || ind__flowX >= in__colsFlow || ind__flowY >= in__rowsFlow || (u == 0 && v == 0)) continue;


            //---------------------------------------------------------------------------------------------
            // calculate number of neighbors
            //---------------------------------------------------------------------------------------------
            float flowX = ((float*)((char*)dinptr__3DFlowX + ind__flowY * in__stepFlow))[ind__flowX];
            float flowY = ((float*)((char*)dinptr__3DFlowY + ind__flowY * in__stepFlow))[ind__flowX];
            if (flowX == 0 && flowY == 0) continue;


            //---------------------------------------------------------------------------------------------
            // calculate weights
            //---------------------------------------------------------------------------------------------
            float weight = 0.0f;
            if ((flowX * flowX) + (flowY * flowY) >= in__threshold) weight = 1.0f;
            weights += weight;


            //---------------------------------------------------------------------------------------------
            // calculate SSDs
            //---------------------------------------------------------------------------------------------
            float f = atan2((double)v,(double)u);
            float alpha = atan2((double)flowY,(double) flowX);
            float ssd = ((f - alpha) * (f - alpha)) * weight;
            ssds += ssd;

        }
    }


    //---------------------------------------------------------------------------------------------
    // calculate result SSD
    //---------------------------------------------------------------------------------------------
    if (weights == 0) weights = 1;
    float resultSSD = (1 / weights) * ssds;


    //---------------------------------------------------------------------------------------------
    // save results in array
    //---------------------------------------------------------------------------------------------
    ((float*)((char*)doutptr__SSDs + posWithinFieldY * in__stepSSDs))[posWithinFieldX] = resultSSD;

}

void GlobalMotionKernel::calculateSSDs(cv::gpu::GpuMat &din__3DFlowX, cv::gpu::GpuMat &din__3DFlowY, int startX, int startY, int endX, int endY, int in__w, float in__threshold, cv::gpu::GpuMat &dout__SSDs) {
    int cols = din__3DFlowX.cols;
    int rows = din__3DFlowX.rows;
    int frameSize = (endX - startX + 1) * (endY - startY + 1);
    int threadSize = 1024;
    int blockSize = (frameSize / threadSize) + 1;
    if (blockSize <= 0) blockSize = 1;

    dv__iterateOverTheWholeImage<<<blockSize, threadSize>>>(din__3DFlowX.ptr<float>(), din__3DFlowY.ptr<float>(), din__3DFlowX.step, rows, cols, startX, startY, endX, endY, in__w, in__threshold, dout__SSDs.step, dout__SSDs.ptr<float>());
}




//---------------------------------------------------------------------------------------------
// GET POSITION OF MIN SSD
//---------------------------------------------------------------------------------------------

__global__ void dv__fillIdxs(int *dinptr__minIdxs, int size) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= size) return;


    //---------------------------------------------------------------------------------------------
    // fill every field with its idx
    //---------------------------------------------------------------------------------------------
    dinptr__minIdxs[idx] = idx;

}

__global__ void dv__getPositionOfMinSSD(float *dinptr__resultSSDs, int *dinptr__minIdxs, size_t in__step, int in__cols, int in__size) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx__SSDLeft = dinptr__minIdxs[idx];
    int y = idx__SSDLeft / in__cols;
    int x = idx__SSDLeft % in__cols;
    int halfsize = in__size / 2;
    bool isOdd = (in__size % 2 == 1);


    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= halfsize) return;


    //---------------------------------------------------------------------------------------------
    // get position of value that should be summed up
    //---------------------------------------------------------------------------------------------
    int idx__SSDRight = dinptr__minIdxs[idx + halfsize];
    int yRight = idx__SSDRight / in__cols;
    int xRight = idx__SSDRight % in__cols;


    //---------------------------------------------------------------------------------------------
    // sum up values of left and right half and save result in left half
    //---------------------------------------------------------------------------------------------
    if (idx == halfsize - 1 && isOdd) {
        int idx__SSDRightLast = dinptr__minIdxs[idx + halfsize + 1];
        int yRightLast = idx__SSDRightLast / in__cols;
        int xRightLast = idx__SSDRightLast % in__cols;

        //---------------------------------------------------------------------------------------------
        // get idx smallest SSD
        //---------------------------------------------------------------------------------------------
        float lastSSDOfLeftHalf = ((float*)((char*)dinptr__resultSSDs + y * in__step))[x];
        float secondLastSSDOfRightHalf = ((float*)((char*)dinptr__resultSSDs + yRight * in__step))[xRight];
        float lastSSDOfRightHalf = ((float*)((char*)dinptr__resultSSDs + yRightLast * in__step))[xRightLast];

        int minIdx = (lastSSDOfLeftHalf < secondLastSSDOfRightHalf) ? idx__SSDLeft : idx__SSDRight;
        float tempSSD = (lastSSDOfLeftHalf < secondLastSSDOfRightHalf) ? lastSSDOfLeftHalf : secondLastSSDOfRightHalf;
        minIdx = (tempSSD < lastSSDOfRightHalf) ? minIdx : idx__SSDRightLast;
        dinptr__minIdxs[idx] = minIdx;

    }else {

        //---------------------------------------------------------------------------------------------
        // get idx smallest SSD
        //---------------------------------------------------------------------------------------------
        float SSDOfLeftHalf = ((float*)((char*)dinptr__resultSSDs + y * in__step))[x];
        float SSDOfRightHalf = ((float*)((char*)dinptr__resultSSDs + yRight * in__step))[xRight];

        int minIdx = (SSDOfLeftHalf < SSDOfRightHalf) ? idx__SSDLeft : idx__SSDRight;
        dinptr__minIdxs[idx] = minIdx;

    }

}

void GlobalMotionKernel::getPositionOfMinSSD(cv::gpu::GpuMat din__resultSSDs, int &out__xMinSSD, int &out__yMinSSD) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int cols = din__resultSSDs.cols;
    int rows = din__resultSSDs.rows;
    int size = rows * cols;
    int threadSize = 1024;
    int blockSize = (size / threadSize) + 1;
    if (blockSize <= 0) blockSize = 1;


    cv::Mat ssdMat;
    float minVal = 100.0f;
    int minX = 0;
    int minY = 0;
    int counter = 0;
    din__resultSSDs.download(ssdMat);
    for (int j = 0; j < ssdMat.rows; j++) {
        for (int i = 0; i < ssdMat.cols; i++) {
            float val = ssdMat.at<float>(j,i);
            if (val < minVal) {
                minVal = val;
                minX = i;
                minY = j;
            }
            if (val == -1) {
                counter++;
            }
        }
    }
//    std::cout << "Smallest value: " << minVal << "at x: " << minX << ", y: " << minY << std::endl;
//    std::cout << counter << " Values are -1" << std::endl;


    //---------------------------------------------------------------------------------------------
    // fill idxs
    //---------------------------------------------------------------------------------------------
    int *ptr__minIdxs = new int[size];
    std::fill_n(ptr__minIdxs, size, 0);
    int *dptr__minIdxs;
    cudaMalloc((void**)&dptr__minIdxs, sizeof(int) * size);
    cudaMemcpy(dptr__minIdxs, ptr__minIdxs, sizeof(int) * size, cudaMemcpyHostToDevice);

    dv__fillIdxs<<<blockSize, threadSize>>>(dptr__minIdxs, size);
    cudaDeviceSynchronize();


    //---------------------------------------------------------------------------------------------
    // sum up data by splitting GpuMat in 2 parts and sum up left and right part,
    // save result in the left half and then split the left half in 2 parts and repeat the same
    // for those two parts until the size of the left have is 1
    //---------------------------------------------------------------------------------------------
    int tempSize = size;
    while (tempSize > 1) {
        dv__getPositionOfMinSSD<<<blockSize, threadSize>>>(din__resultSSDs.ptr<float>(), dptr__minIdxs, din__resultSSDs.step, cols, tempSize);
        cudaDeviceSynchronize();
        tempSize = tempSize / 2;
    }


    //---------------------------------------------------------------------------------------------
    // copy back results
    //---------------------------------------------------------------------------------------------
    int *ptr__resultIdx = new int;
    cudaMemcpy(ptr__resultIdx, dptr__minIdxs, sizeof(int), cudaMemcpyDeviceToHost);
    int minIdx = *ptr__resultIdx;
    out__yMinSSD = minIdx / cols;
    out__xMinSSD = minIdx % cols;

//    std::cout << "gpu smallest value: at x: " << out__xMinSSD << ", y: " << out__yMinSSD << std::endl;
//    std::cout << "Frame dimensions are: x: " << din__resultSSDs.cols - 1 << " y: " << din__resultSSDs.rows - 1 << std::endl;

}




//---------------------------------------------------------------------------------------------
// create synthetic flow field
//---------------------------------------------------------------------------------------------

__global__ void dv__createSyntheticFlowField(float *ex__syntheticFlowFieldX, float *ex__syntheticFlowFieldY, size_t in__step, int in__cols, int in__rows, int foeX, int foeY, float maxRotation, float translationX, float translationY) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float u1 = 0;
    float u2 = 0;
    float u3 = maxRotation;

    int yLeft = foeY;
    int yRight = (in__rows - 1) - foeY;
    int yMax = (yLeft > yRight) ? yLeft : yRight;
    int xLeft = foeX;
    int xRight = (in__cols - 1) - foeX;
    int xMax = (xLeft > xRight) ? xLeft : xRight;

    int y = (idx / in__cols) - yLeft;
    int x = (idx % in__cols) - xLeft;


    //---------------------------------------------------------------------------------------------
    // calculate vector
    //---------------------------------------------------------------------------------------------
    float v1 = (float)x / xMax;
    float v2 = (float)y / yMax;
    float v3 = 0;
    float s1 = (u2 * v3) - (u3 * v2);
    float s2 = (u3 * v1) - (u1 * v3);
    float s3 = (u1 * v2) - (u2 * v1);


    //---------------------------------------------------------------------------------------------
    // copy back result
    //---------------------------------------------------------------------------------------------
    ((float*)((char*)ex__syntheticFlowFieldX + (y + foeY) * in__step))[x + foeX] = s1 + translationX;
    ((float*)((char*)ex__syntheticFlowFieldY + (y + foeY) * in__step))[x + foeX] = s2 + translationY;

}

void GlobalMotionKernel::createSyntheticFlowField(cv::gpu::GpuMat &ex__syntheticFlowFieldX, cv::gpu::GpuMat &ex__syntheticFlowFieldY, int foeX, int foeY, float maxRotation, float translationX, float translationY) {

    int cols = ex__syntheticFlowFieldX.cols;
    int rows = ex__syntheticFlowFieldX.rows;
    int size = cols * rows;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size / threadsPerBlock) + 1;
    dv__createSyntheticFlowField<<<blocksPerGrid, threadsPerBlock>>>(ex__syntheticFlowFieldX.ptr<float>(), ex__syntheticFlowFieldY.ptr<float>(), ex__syntheticFlowFieldX.step, cols, rows, foeX, foeY, maxRotation, translationX, translationY);
}




//---------------------------------------------------------------------------------------------
// find best mathing flowfield
//---------------------------------------------------------------------------------------------

__global__ void dv__calculateDivergence(float *dinptr__syntheticFlowFieldX, float *dinptr__syntheticFlowFieldY, float *dinptr__realFlowFieldX, float *dinptr__realFlowFieldY, size_t in__step, int in__cols, int in__rows, float *dout__divergences) {

    //---------------------------------------------------------------------------------------------
    // shared temp memory
    //---------------------------------------------------------------------------------------------
    __shared__ float cache[256];


    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int size = in__cols  * in__rows;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % in__cols;
    int y = idx / in__cols;
    int idx__cache = threadIdx.x;


    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= size) return;


    //---------------------------------------------------------------------------------------------
    // calculate divergences
    //---------------------------------------------------------------------------------------------
    float syntheticFlowX = ((float*)((char*)dinptr__syntheticFlowFieldX + y * in__step))[x];
    float syntheticFlowY = ((float*)((char*)dinptr__syntheticFlowFieldY + y * in__step))[x];
    float realFlowX = ((float*)((char*)dinptr__realFlowFieldX + y * in__step))[x];
    float realFlowY = ((float*)((char*)dinptr__realFlowFieldY + y * in__step))[x];
    float divergence = (syntheticFlowX - realFlowX) * (syntheticFlowX - realFlowX) + (syntheticFlowY - realFlowY) * (syntheticFlowY - realFlowY);


    //---------------------------------------------------------------------------------------------
    // fill cache and wait until all threads within this block did so
    //---------------------------------------------------------------------------------------------
    cache[idx__cache] = divergence;
    __syncthreads();


    //---------------------------------------------------------------------------------------------
    // data reduction
    //---------------------------------------------------------------------------------------------
    int i = blockDim.x / 2;
    while (i != 0) {
        if (idx__cache < i) {
            cache[idx__cache] += cache[idx__cache + i];
        }
        __syncthreads();
        i /= 2;
    }


    //---------------------------------------------------------------------------------------------
    // one thread need to save the accumulated result in the global memory
    //---------------------------------------------------------------------------------------------
    if (idx__cache == 0) {
        dout__divergences[blockIdx.x] = cache[0];
    }

}

void GlobalMotionKernel::calculateDivergenceOfFlowFields(std::vector<cv::gpu::GpuMat> &in__syntheticFlowFields, cv::gpu::GpuMat &in__realFlowFieldX, cv::gpu::GpuMat &in__realFlowFieldY, int &out__idxOfBestMatch) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int cols = in__realFlowFieldX.cols;
    int rows = in__realFlowFieldX.rows;
    int size = cols * rows;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size / threadsPerBlock) + 1;

    float *realFlowX = in__realFlowFieldX.ptr<float>();
    float *realFlowY = in__realFlowFieldY.ptr<float>();
    float *ptr__divergences = new float[blocksPerGrid];
    float *dptr__divergences;
    cudaMalloc((void**)&dptr__divergences, sizeof(float) * blocksPerGrid);
    cudaMemcpy(dptr__divergences, ptr__divergences, sizeof(float) * blocksPerGrid, cudaMemcpyHostToDevice);

    int numberOfFlowFields = in__syntheticFlowFields.size() / 2;


    //---------------------------------------------------------------------------------------------
    // find index of synthetic flow field with smallest divergence to the real flow field
    //---------------------------------------------------------------------------------------------
    int smallestIdx = -1;
    float smallestDivergence = std::numeric_limits<float>::max();
    for (int ind = 0; ind < numberOfFlowFields; ind+=2) {
        float *syntheticFlowX = in__syntheticFlowFields.at(ind).ptr<float>();
        float *syntheticFlowY = in__syntheticFlowFields.at(ind+1).ptr<float>();
        dv__calculateDivergence<<<blocksPerGrid, threadsPerBlock>>>(syntheticFlowX, syntheticFlowY, realFlowX, realFlowY, in__realFlowFieldX.step, in__realFlowFieldX.cols, in__realFlowFieldX.rows, dptr__divergences);
        cudaMemcpy(ptr__divergences, dptr__divergences, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost);
        float divergence = 0;
        for (int i = 0; i < blocksPerGrid; i++) {
            divergence += ptr__divergences[i];
        }
        if(divergence < smallestDivergence) {
            smallestDivergence = divergence;
            smallestIdx = ind;
        }
    }


    //---------------------------------------------------------------------------------------------
    // return smallest index
    //---------------------------------------------------------------------------------------------
    out__idxOfBestMatch = smallestIdx;


    //---------------------------------------------------------------------------------------------
    // free memory
    //---------------------------------------------------------------------------------------------
    free(ptr__divergences);
    cudaFree(dptr__divergences);
}

