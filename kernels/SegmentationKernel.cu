#include "SegmentationKernel.h"



//---------------------------------------------------------------------------------------------------------------------------------
// Mean
//---------------------------------------------------------------------------------------------------------------------------------
__global__ void dv__calcMean(
        int *inptr__classes,
        size_t stepInt,
        float *inptr__YChannel, float *inptr__UChannel, float *inptr__VChannel,
        float *inptr__uFlow, float *inptr__vFLow,
        size_t step,
        int classK,
        int cols, int rows,
        float *outptr__xPosMean, float *outptr__yPosMean,
        float *outptr__YChMean, float *outptr__UChMean, float *outptr__VChMean,
        float *outptr__uFlowMean, float *outptr__vFlowMean) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idx__cache = threadIdx.x;
    int size = rows * cols;
    int x = idx % cols;
    int y = idx / cols;


    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= size) return;



    //---------------------------------------------------------------------------------------------
    // setup caches
    //---------------------------------------------------------------------------------------------
    __shared__ float cache_XPos[256];
    __shared__ float cache_YPos[256];
    __shared__ float cache_YChannel[256];
    __shared__ float cache_UChannel[256];
    __shared__ float cache_VChannel[256];
    __shared__ float cache_uFlow[256];
    __shared__ float cache_vFlow[256];



    //---------------------------------------------------------------------------------------------
    // fill caches
    //---------------------------------------------------------------------------------------------
    int k = ((int*)(inptr__classes + y*stepInt))[x];
    cache_XPos[idx__cache] = (k == classK) ? x : 0.0f;
    cache_YPos[idx__cache] = (k == classK) ? y : 0.0f;
    cache_YChannel[idx__cache] = (k == classK) ? ((float*)((char*)inptr__YChannel + y*step))[x] : 0.0f;
    cache_UChannel[idx__cache] = (k == classK) ? ((float*)((char*)inptr__UChannel + y*step))[x] : 0.0f;
    cache_VChannel[idx__cache] = (k == classK) ? ((float*)((char*)inptr__VChannel + y*step))[x] : 0.0f;
    cache_uFlow[idx__cache] = (k == classK) ? ((float*)((char*)inptr__uFlow + y*step))[x] : 0.0f;
    cache_vFlow[idx__cache] = (k == classK) ? ((float*)((char*)inptr__vFLow + y*step))[x] : 0.0f;
    __syncthreads();



    //---------------------------------------------------------------------------------------------
    // data reduction
    //---------------------------------------------------------------------------------------------
    int i = blockDim.x / 2;
    while (i != 0) {
        if (idx__cache < i) {
            cache_XPos[idx__cache] += cache_XPos[idx__cache + i];
            cache_YPos[idx__cache] += cache_YPos[idx__cache + i];
            cache_YChannel[idx__cache] += cache_YChannel[idx__cache + i];
            cache_UChannel[idx__cache] += cache_UChannel[idx__cache + i];
            cache_VChannel[idx__cache] += cache_VChannel[idx__cache + i];
            cache_uFlow[idx__cache] += cache_uFlow[idx__cache + i];
            cache_vFlow[idx__cache] += cache_vFlow[idx__cache + i];
        }
        __syncthreads();
        i /= 2;
    }


    //---------------------------------------------------------------------------------------------
    // one thread need to save the accumulated result in the global memory
    //---------------------------------------------------------------------------------------------
    if (idx__cache == 0) {
        outptr__xPosMean[blockIdx.x] = cache_XPos[0];
        outptr__yPosMean[blockIdx.x] = cache_YPos[0];
        outptr__YChMean[blockIdx.x] = cache_YChannel[0];
        outptr__UChMean[blockIdx.x] = cache_UChannel[0];
        outptr__VChMean[blockIdx.x] = cache_VChannel[0];
        outptr__uFlowMean[blockIdx.x] = cache_uFlow[0];
        outptr__vFlowMean[blockIdx.x] = cache_vFlow[0];
    }

}




void SegmentationKernel::calcMean(
        cv::gpu::GpuMat &in__classes,
        cv::gpu::GpuMat &in__YChannel, cv::gpu::GpuMat &in__UChannel, cv::gpu::GpuMat &in__VChannel,
        cv::gpu::GpuMat &in__uFLow, cv::gpu::GpuMat &in__vFlow,
        int in__classK,
        int in__numberOfPoints,
        float *outptr__meanVector) {

    //---------------------------------------------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------------------------------------------
    int size = in__classes.cols * in__classes.rows;
    int threadSize = 256;
    int blockSize = (size / threadSize) + 1;



    //---------------------------------------------------------------------------------------------------------------------------------
    // copy back results
    //---------------------------------------------------------------------------------------------------------------------------------
    float *ptr__xPosMean = new float[blockSize];
    float *ptr__yPosMean = new float[blockSize];
    float *ptr__YChannelMean = new float[blockSize];
    float *ptr__UChannelMean = new float[blockSize];
    float *ptr__VChannelMean = new float[blockSize];
    float *ptr__uFlowMean = new float[blockSize];
    float *ptr__vFlowMean = new float[blockSize];
    std::fill_n(ptr__xPosMean, blockSize, 0.0f);
    std::fill_n(ptr__yPosMean, blockSize, 0.0f);
    std::fill_n(ptr__YChannelMean, blockSize, 0.0f);
    std::fill_n(ptr__UChannelMean, blockSize, 0.0f);
    std::fill_n(ptr__VChannelMean, blockSize, 0.0f);
    std::fill_n(ptr__uFlowMean, blockSize, 0.0f);
    std::fill_n(ptr__vFlowMean, blockSize, 0.0f);
    float *dptr__xPosMean;
    float *dptr__yPosMean;
    float *dptr__YChannelMean;
    float *dptr__UChannelMean;
    float *dptr__VChannelMean;
    float *dptr__uFlowMean;
    float *dptr__vFlowMean;
    cudaMalloc((void**)&dptr__xPosMean, sizeof(float) * blockSize);
    cudaMalloc((void**)&dptr__yPosMean, sizeof(float) * blockSize);
    cudaMalloc((void**)&dptr__YChannelMean, sizeof(float) * blockSize);
    cudaMalloc((void**)&dptr__UChannelMean, sizeof(float) * blockSize);
    cudaMalloc((void**)&dptr__VChannelMean, sizeof(float) * blockSize);
    cudaMalloc((void**)&dptr__uFlowMean, sizeof(float) * blockSize);
    cudaMalloc((void**)&dptr__vFlowMean, sizeof(float) * blockSize);
    cudaMemcpy(dptr__xPosMean, ptr__xPosMean, sizeof(float) * blockSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dptr__yPosMean, ptr__yPosMean, sizeof(float) * blockSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dptr__YChannelMean, ptr__YChannelMean, sizeof(float) * blockSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dptr__UChannelMean, ptr__UChannelMean, sizeof(float) * blockSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dptr__VChannelMean, ptr__VChannelMean, sizeof(float) * blockSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dptr__uFlowMean, ptr__uFlowMean, sizeof(float) * blockSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dptr__vFlowMean, ptr__vFlowMean, sizeof(float) * blockSize, cudaMemcpyHostToDevice);



    //---------------------------------------------------------------------------------------------------------------------------------
    // calculate mean in kernel
    //---------------------------------------------------------------------------------------------------------------------------------
    dv__calcMean<<<blockSize, threadSize>>>(
                 in__classes.ptr<int>(),
                 in__classes.step,
                 in__YChannel.ptr<float>(), in__UChannel.ptr<float>(), in__VChannel.ptr<float>(),
                 in__uFLow.ptr<float>(), in__vFlow.ptr<float>(),
                 in__YChannel.step,
                 in__classK,
                 in__YChannel.cols, in__YChannel.rows,
                 dptr__xPosMean, dptr__yPosMean,
                 dptr__YChannelMean, dptr__UChannelMean, dptr__VChannelMean,
                 dptr__uFlowMean, dptr__vFlowMean);



    //---------------------------------------------------------------------------------------------------------------------------------
    // copy back results
    //---------------------------------------------------------------------------------------------------------------------------------
    cudaMemcpy(ptr__xPosMean, dptr__xPosMean, sizeof(float) * blockSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ptr__yPosMean, dptr__yPosMean, sizeof(float) * blockSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ptr__YChannelMean, dptr__YChannelMean, sizeof(float) * blockSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ptr__UChannelMean, dptr__UChannelMean, sizeof(float) * blockSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ptr__VChannelMean, dptr__VChannelMean, sizeof(float) * blockSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ptr__uFlowMean, dptr__uFlowMean, sizeof(float) * blockSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ptr__vFlowMean, dptr__vFlowMean, sizeof(float) * blockSize, cudaMemcpyDeviceToHost);



    //---------------------------------------------------------------------------------------------------------------------------------
    // sum up the rest of the arrays
    //---------------------------------------------------------------------------------------------------------------------------------
    float xPosMean = 0.0f;
    float yPosMean = 0.0f;
    float YChannelMean = 0.0f;
    float UChannelMean = 0.0f;
    float VChannelMean = 0.0f;
    float uFlowMean = 0.0f;
    float vFlowMean = 0.0f;

    for (int i = 0; i < blockSize; i++) {
        xPosMean += ptr__xPosMean[i];
        yPosMean += ptr__yPosMean[i];
        YChannelMean += ptr__YChannelMean[i];
        UChannelMean += ptr__UChannelMean[i];
        VChannelMean += ptr__VChannelMean[i];
        uFlowMean += ptr__uFlowMean[i];
        vFlowMean += ptr__vFlowMean[i];
    }



    //---------------------------------------------------------------------------------------------------------------------------------
    // return mean
    //---------------------------------------------------------------------------------------------------------------------------------

    printf("number of points is %d \n", in__numberOfPoints );

    outptr__meanVector[0] = xPosMean / in__numberOfPoints;
    outptr__meanVector[1] = yPosMean / in__numberOfPoints;
    outptr__meanVector[2] = YChannelMean / in__numberOfPoints;
    outptr__meanVector[3] = UChannelMean / in__numberOfPoints;
    outptr__meanVector[4] = VChannelMean / in__numberOfPoints;
    outptr__meanVector[5] = uFlowMean / in__numberOfPoints;
    outptr__meanVector[6] = vFlowMean / in__numberOfPoints;



    //---------------------------------------------------------------------------------------------------------------------------------
    // release memory
    //---------------------------------------------------------------------------------------------------------------------------------
    cudaFree(dptr__xPosMean);
    cudaFree(dptr__yPosMean);
    cudaFree(dptr__YChannelMean);
    cudaFree(dptr__UChannelMean);
    cudaFree(dptr__VChannelMean);
    cudaFree(dptr__uFlowMean);
    cudaFree(dptr__vFlowMean);
    free(ptr__xPosMean);
    free(ptr__yPosMean);
    free(ptr__YChannelMean);
    free(ptr__UChannelMean);
    free(ptr__VChannelMean);
    free(ptr__uFlowMean);
    free(ptr__vFlowMean);

}






//---------------------------------------------------------------------------------------------------------------------------------
// Covarianz matrix
//---------------------------------------------------------------------------------------------------------------------------------
__global__ void dv__calculateDeviationMatrix(
        int *inptr__classes,
        size_t stepInt,
        float *inptr__YChannel, float *inptr__UChannel, float *inptr__VChannel,
        float *inptr__uFlow, float *inptr__vFLow,
        float *inptr__meanVector,
        size_t step,
        int cols, int rows,
        int classK,
        float *outptr__deviationMatrix,
        size_t stepDev) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int size = rows * cols;
    int x = idx % cols;
    int y = idx / cols;


    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= size) return;



    //---------------------------------------------------------------------------------------------
    // fill matrix
    //---------------------------------------------------------------------------------------------
    float xPos = (float)x - inptr__meanVector[0];
    float yPos = (float)y - inptr__meanVector[1];
    float yChannel = ((float*)((char*)inptr__YChannel + y*step))[x] - inptr__meanVector[2];
    float uChannel = ((float*)((char*)inptr__UChannel + y*step))[x] - inptr__meanVector[3];
    float vChannel = ((float*)((char*)inptr__VChannel + y*step))[x] - inptr__meanVector[4];
    float uFlow = ((float*)((char*)inptr__uFlow + y*step))[x] - inptr__meanVector[5];
    float vFlow = ((float*)((char*)inptr__vFLow + y*step))[x] - inptr__meanVector[6];
    int k = ((int*)((char*)inptr__classes + y*stepInt))[x];


    if (k == classK) {
        ((float*)((char*)outptr__deviationMatrix + 0*stepDev))[0] = xPos;
        ((float*)((char*)outptr__deviationMatrix + idx*stepDev))[1] = yPos;
        ((float*)((char*)outptr__deviationMatrix + idx*stepDev))[2] = yChannel;
        ((float*)((char*)outptr__deviationMatrix + idx*stepDev))[3] = uChannel;
        ((float*)((char*)outptr__deviationMatrix + idx*stepDev))[4] = vChannel;
        ((float*)((char*)outptr__deviationMatrix + idx*stepDev))[5] = uFlow;
        ((float*)((char*)outptr__deviationMatrix + idx*stepDev))[6] = vFlow;
    }

}


__global__ void dv__fillTempMat(float *matA, size_t step, int cols, int rows, float *tempMat, size_t stepTemp) {

        //---------------------------------------------------------------------------------------------
        // setup variables
        //---------------------------------------------------------------------------------------------
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int size = cols * cols * rows;
        int n = cols * rows;
        int tempX = idx / rows;
        int tempY = idx % rows;
        int x1 = idx / n;
        int y1 = idx % rows;
        int x2 = (idx / rows) % cols;
        int y2 = idx % rows;



        //---------------------------------------------------------------------------------------------
        // bounds check
        //---------------------------------------------------------------------------------------------
        if (idx > size) return;



        //---------------------------------------------------------------------------------------------
        // fill values
        //---------------------------------------------------------------------------------------------
        float val1 = ((float*)((char*)matA + y1*step))[x1];
        float val2 = ((float*)((char*)matA + y2*step))[x2];
        ((float*)((char*)tempMat + tempY*stepTemp))[tempX] = val1 * val2;

}

__global__ void dv__sumupValues(float *matA, size_t step, int size, int cols, int rows) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int halfSize = size / 2;
    int sumUpArea = halfSize * cols;
    int x = idx / halfSize;
    int y = idx % halfSize;
    bool isOdd = (halfSize * 2 != size);



    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= sumUpArea) return;



    //---------------------------------------------------------------------------------------------
    // sum up values
    //---------------------------------------------------------------------------------------------
    float result = 0.0f;

    float val1 = ((float*)((char*)matA + y*step))[x];
    int y2 = y + halfSize;
    float val2 = ((float*)((char*)matA + y2*step))[x];
    result = val1 + val2;

    if (y == halfSize - 1 && isOdd) {
        int y3 = y2 + 1;
        float val3 = ((float*)((char*)matA + y3*step))[x];
        result += val3;
    }



    //---------------------------------------------------------------------------------------------
    // save result
    //---------------------------------------------------------------------------------------------
    ((float*)((char*)matA + y*step))[x] = result;

}



__global__ void dv__createCovarianzMatrix(float *results, size_t step, int cols, float n, float *covarianzMat, size_t stepCov, int colsCov) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int x = idx % colsCov;
    int y = idx / colsCov;



    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= cols) return;



    //---------------------------------------------------------------------------------------------
    // create cov
    //---------------------------------------------------------------------------------------------
    float val = ((float*)((char*)results + 0*step))[idx];
    val /= n;
    ((float*)((char*)covarianzMat + y*stepCov))[x] = val;

}



void SegmentationKernel::calcCovarianzMatrix(
        cv::gpu::GpuMat &in__classes,
        cv::gpu::GpuMat &in__YChannel, cv::gpu::GpuMat &in__UChannel, cv::gpu::GpuMat &in__VChannel,
        cv::gpu::GpuMat &in__uFLow, cv::gpu::GpuMat &in__vFlow,
        float *inptr__meanVector,
        int in__classK,
        int in__numberOfPoints,
        cv::gpu::GpuMat &out__covarianzMatrix) {

    //---------------------------------------------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------------------------------------------
    int size = in__YChannel.cols * in__YChannel.rows;
    int threadSize = 1024;
    int blockSize = (size / threadSize) + 1;


    //---------------------------------------------------------------------------------------------------------------------------------
    // transfer mean vector to gpu
    //---------------------------------------------------------------------------------------------------------------------------------
    float *dptr__meanVector;
    cudaMalloc((void**)&dptr__meanVector, sizeof(float) * 7);
    cudaMemcpy(dptr__meanVector, inptr__meanVector, sizeof(float) * 7, cudaMemcpyHostToDevice);



    //---------------------------------------------------------------------------------------------------------------------------------
    // calculate covarianz matrix
    //---------------------------------------------------------------------------------------------------------------------------------
    // used: http://stattrek.com/matrix-algebra/covariance-matrix.aspx
    // deviation matrix: A = X - X_mean
    cv::gpu::GpuMat d__A(size, 7, CV_32FC1, cv::Scalar(0.0f));
    dv__calculateDeviationMatrix<<<blockSize, threadSize>>>(
                in__classes.ptr<int>(),
                in__classes.step,
                in__YChannel.ptr<float>(), in__UChannel.ptr<float>(), in__VChannel.ptr<float>(),
                in__uFLow.ptr<float>(), in__vFlow.ptr<float>(),
                dptr__meanVector,
                in__YChannel.step,
                in__YChannel.cols, in__YChannel.rows,
                in__classK,
                d__A.ptr<float>(),
                d__A.step);

//    cv::Mat temp1;
//    d__A.download(temp1);
//    std::cout << "deviation mat: \n" << temp1 << std::endl;

    cv::gpu::GpuMat intermediateMat(d__A.rows, d__A.cols * d__A.cols, CV_32FC1, cv::Scalar(0.0f));
    size = intermediateMat.cols * intermediateMat.rows;
    blockSize = (size / threadSize) + 1;
    dv__fillTempMat<<<blockSize, threadSize>>>(d__A.ptr<float>(), d__A.step, d__A.cols, d__A.rows, intermediateMat.ptr<float>(), intermediateMat.step);

//    cv::Mat temp2;
//    intermediateMat.download(temp2);
//    std::cout << "inter 1 mat: \n" << temp2 << std::endl;

    int tempsize = intermediateMat.rows;
    while (tempsize != 0) {
        dv__sumupValues<<<blockSize, threadSize>>>(intermediateMat.ptr<float>(), intermediateMat.step, tempsize, intermediateMat.cols, intermediateMat.rows);
        cudaDeviceSynchronize();
        tempsize /= 2;
    }

//    cv::Mat temp3;
//    intermediateMat.download(temp3);
//    std::cout << "inter 2 mat: \n" << temp3 << std::endl;

    size = intermediateMat.cols;
    blockSize = (size / threadSize) + 1;
    dv__createCovarianzMatrix<<<blockSize, threadSize>>>(intermediateMat.ptr<float>(), intermediateMat.step, intermediateMat.cols, in__numberOfPoints, out__covarianzMatrix.ptr<float>(), out__covarianzMatrix.step, out__covarianzMatrix.cols);
}




//---------------------------------------------------------------------------------------------------------------------------------
// Segmentation
//---------------------------------------------------------------------------------------------------------------------------------
__global__ void dv__calculateFlowAndColorLikelihood(
        float *inptr__YChannel, float *inptr__UChannel, float *inptr__VChannel,
        float *inptr__uFlow, float *inptr__vFlow,
        size_t stepFrame,
        int rows, int cols,
        float *inptr__covarianceMatrix,
        size_t stepCov,
        float *inptr__means,
        float in_PI,
        float *outptr__flowLogLikelihoods, float *outptr__colorLogLikelihoods,
        float *exptr__maxFlowLogLikelihoods) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int size = rows * cols;
    int x = idx % cols;
    int y = idx / cols;



    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= size) return;



    //---------------------------------------------------------------------------------------------
    // calculate probability density functions
    //---------------------------------------------------------------------------------------------
    //calculate color PDF
    float a = ((float*)((char*)inptr__covarianceMatrix + 2*stepCov))[2];
    float b = ((float*)((char*)inptr__covarianceMatrix + 2*stepCov))[3];
    float c = ((float*)((char*)inptr__covarianceMatrix + 2*stepCov))[4];
    float d = ((float*)((char*)inptr__covarianceMatrix + 3*stepCov))[2];
    float e = ((float*)((char*)inptr__covarianceMatrix + 3*stepCov))[3];
    float f = ((float*)((char*)inptr__covarianceMatrix + 3*stepCov))[4];
    float g = ((float*)((char*)inptr__covarianceMatrix + 4*stepCov))[2];
    float h = ((float*)((char*)inptr__covarianceMatrix + 4*stepCov))[3];
    float i = ((float*)((char*)inptr__covarianceMatrix + 4*stepCov))[4];
    float k = ((float*)((char*)inptr__YChannel + y*stepFrame))[x] - inptr__means[2];
    float l = ((float*)((char*)inptr__UChannel + y*stepFrame))[x] - inptr__means[3];
    float m = ((float*)((char*)inptr__VChannel + y*stepFrame))[x] - inptr__means[4];

    float part1 = k * ( k*(e*i-f*h) + l*(c*h-b*i) + m*(b*f-c*e) );
    float part2 = l * ( k*(f*g-d*i) + l*(a*i-c*g) + m*(c*d-a*f) );
    float part3 = m * ( k*(d*h-e*g) + l*(b*g-a*h) + m*(a*e-b*d) );
    float colorDeterminant = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g);
    float colorExponent = 1.0f;
    float colorPDF = 0.0f;
    if (colorDeterminant != 0) {
        colorExponent = expf( -1.0f * ((part1 + part2 + part3) / (2 * colorDeterminant)) );
        colorPDF = (1 / (powf((2 * in_PI), 3.0f/2) * powf(colorDeterminant, 1.0f/2)) * colorExponent);
    }

    // calculate flow PDF
    a = ((float*)((char*)inptr__covarianceMatrix + 5*stepCov))[5];
    b = ((float*)((char*)inptr__covarianceMatrix + 5*stepCov))[6];
    c = ((float*)((char*)inptr__covarianceMatrix + 6*stepCov))[5];
    d = ((float*)((char*)inptr__covarianceMatrix + 6*stepCov))[6];
    e = ((float*)((char*)inptr__uFlow + y*stepFrame))[x] - inptr__means[5];
    f = ((float*)((char*)inptr__vFlow + y*stepFrame))[x] - inptr__means[6];

    float flowDeterminant = a*d - b*c;
    float flowExponent = 1.0;
    float flowPDF = 0.0f;
    if (flowDeterminant != 0) {
        flowDeterminant = expf( -1 * ((e*(d*e-b*f) + f*(a*f-c*e)) / (2 * flowDeterminant)) );
        flowPDF = (1 / (2 * in_PI * sqrtf(flowDeterminant)) * flowExponent);
    }


    //---------------------------------------------------------------------------------------------
    // get bigger flow
    //---------------------------------------------------------------------------------------------
    float maxFlowLogLikelihood = ((float*)((char*)exptr__maxFlowLogLikelihoods + y*stepFrame))[x];
    if (flowPDF > maxFlowLogLikelihood) {
        ((float*)((char*)exptr__maxFlowLogLikelihoods + y*stepFrame))[x] = flowPDF;
    }



    //---------------------------------------------------------------------------------------------
    // calculate probability density functions
    //---------------------------------------------------------------------------------------------
    float flowLogLikelihood = logf(flowPDF);
    float colorLogLikelihood = logf(colorPDF);
    ((float*)((char*)outptr__flowLogLikelihoods + y*stepFrame))[x] = flowLogLikelihood;
    ((float*)((char*)outptr__colorLogLikelihoods + y*stepFrame))[x] = colorLogLikelihood;

}

void SegmentationKernel::calculateFlowAndColorLikelihood(
        cv::gpu::GpuMat &in__YChannel, cv::gpu::GpuMat &in__UChannel, cv::gpu::GpuMat &in__VChannel,
        cv::gpu::GpuMat &in__uFlow, cv::gpu::GpuMat &in__vFlow,
        cv::gpu::GpuMat &in__covarianceMatrix,
        float *inptr__meanVector,
        cv::gpu::GpuMat &out__flowLogLikelihoods, cv::gpu::GpuMat &out__colorLogLikelihoods,
        cv::gpu::GpuMat &out__maxFlowLikelihoods) {

    //---------------------------------------------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------------------------------------------
    float pi = 3.14159265358979323846f;
    int size = in__YChannel.cols * in__YChannel.rows;
    int threadSize = 1024;
    int blockSize = (size / threadSize) + 1;



    //---------------------------------------------------------------------------------------------------------------------------------
    // transfer mean vector to gpu
    //---------------------------------------------------------------------------------------------------------------------------------
    float *dptr__meanVector;
    cudaMalloc((void**)&dptr__meanVector, sizeof(float) * 7);
    cudaMemcpy(dptr__meanVector, inptr__meanVector, sizeof(float) * 7, cudaMemcpyHostToDevice);



    //---------------------------------------------------------------------------------------------------------------------------------
    // calculate flow and color likelihood
    //---------------------------------------------------------------------------------------------------------------------------------
    dv__calculateFlowAndColorLikelihood<<<blockSize, threadSize>>>(
                in__YChannel.ptr<float>(), in__UChannel.ptr<float>(), in__VChannel.ptr<float>(),
                in__uFlow.ptr<float>(), in__vFlow.ptr<float>(),
                in__YChannel.step,
                in__YChannel.cols, in__YChannel.rows,
                in__covarianceMatrix.ptr<float>(),
                in__covarianceMatrix.step,
                dptr__meanVector,
                pi,
                out__flowLogLikelihoods.ptr<float>(), out__colorLogLikelihoods.ptr<float>(),
                out__maxFlowLikelihoods.ptr<float>());
}


__global__ void dv__makeBinaryImage(int *inptr__classes, size_t step, int cols, int rows, int in__classK, float *outptr__binaryImage) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int size = cols * rows;
    int x = idx % cols;
    int y = idx / cols;



    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= size) return;



    //---------------------------------------------------------------------------------------------
    // calculate value
    //---------------------------------------------------------------------------------------------
    int classK = ((int*)((char*)inptr__classes + y*step))[x];

    if (classK == in__classK) {
        ((float*)((char*)outptr__binaryImage + y*step))[x] = 1.0f;
    }else{
        ((float*)((char*)outptr__binaryImage + y*step))[x] = 0.0f;
    }

}

void SegmentationKernel::makeBinaryImage(cv::gpu::GpuMat &in__classes, int in__classK, cv::gpu::GpuMat &out__binaryImage) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int size = in__classes.cols * in__classes.rows;
    int threadSize = 1024;
    int blocksize = (size / threadSize) + 1;



    //---------------------------------------------------------------------------------------------
    // apply gaussian filter to binary image
    //---------------------------------------------------------------------------------------------
    dv__makeBinaryImage<<<blocksize, threadSize>>>(in__classes.ptr<int>(), in__classes.step, in__classes.cols, in__classes.rows, in__classK, out__binaryImage.ptr<float>());

}





__global__ void dv__matAdd(float *inptr__matA, float *exptr__matB, size_t step, int cols, int rows) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int size = cols * rows;
    int x = idx % cols;
    int y = idx / cols;



    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= size) return;



    //---------------------------------------------------------------------------------------------
    // calculate value
    //---------------------------------------------------------------------------------------------
    float valA = ((float*)((char*)inptr__matA + y*step))[x];
    float valB = ((float*)((char*)exptr__matB + y*step))[x];

    ((float*)((char*)exptr__matB + y*step))[x] = valA + valB;

}

void SegmentationKernel::matAdd(cv::gpu::GpuMat &in__gaussianImage, cv::gpu::GpuMat &ex__spatialDeviations) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int size = in__gaussianImage.cols * in__gaussianImage.rows;
    int threadSize = 1024;
    int blocksize = (size / threadSize) + 1;



    //---------------------------------------------------------------------------------------------
    // sum up values
    //---------------------------------------------------------------------------------------------
    dv__matAdd<<<blocksize, threadSize>>>(in__gaussianImage.ptr<float>(), ex__spatialDeviations.ptr<float>(), in__gaussianImage.step, in__gaussianImage.cols, in__gaussianImage.rows);

}



__global__ void dv__calculateLikelihood(float *inptr__colorLogLikelihoods, float *inptr__flowLogLikelihoods, float *inptr__sumOfSpatialMeans, float *inptr__maxFlowLogLikelihoods, size_t step, int cols, int rows, int numberOfDataPoints, float sigma, int halfSearchRegion,  float in_PI, float *outptr__likelihoods) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int size = cols * rows;
    int x = idx % cols;
    int y = idx / cols;



    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= size) return;



    //---------------------------------------------------------------------------------------------
    // calculate PDFs
    //---------------------------------------------------------------------------------------------
    // calculate spatial PDF
    float spatialMean = inptr__sumOfSpatialMeans[idx];
    float spatialPDF = 1 / (numberOfDataPoints * 2 * in_PI * sigma * sigma) * spatialMean;
    float spatialLogLikelihood = logf(spatialPDF);

    float colorLogLikelihood = ((float*)((char*)inptr__colorLogLikelihoods + y*step))[x];

    float flowLogLikelihood = ((float*)((char*)inptr__flowLogLikelihoods + y*step))[x];



    //---------------------------------------------------------------------------------------------
    // calculate weights
    //---------------------------------------------------------------------------------------------
    float maxFlowLikelihood = inptr__flowLogLikelihoods[idx];
    float maxFlowLikelihoodOfANeighbor = ((float*)((char*)inptr__maxFlowLogLikelihoods + y*step))[x];
    float flowLikelihood = 0;
    for (int j = -halfSearchRegion; j <= halfSearchRegion; j++) {
        for (int i = -halfSearchRegion; i <= halfSearchRegion; i++) {
            if ( (i != 0 || j != 0) && (j + y) >= 0 && (j + y) < rows && (i + x) >= 0 && (i + x) < cols ) {
                int idx__neighbor = (j + y) * cols + (i + x);
                flowLikelihood = inptr__flowLogLikelihoods[idx__neighbor];
                if (flowLikelihood > maxFlowLikelihoodOfANeighbor) {
                    maxFlowLikelihoodOfANeighbor = flowLikelihood;
                }
            }
        }
    }

    float a = 0.5f;
    float rho1 = 1 + expf((-a) * inptr__flowLogLikelihoods[idx]);

    float shift = 1;             // TODO FIND A SUITEABLE VALUE
    float d = fabsf(maxFlowLikelihood - maxFlowLikelihoodOfANeighbor);
    float rho2 = 1 + expf((-a) * (d - shift));

    float weightFlow = rho1 * rho2;
    float weightColor = 1 - (weightFlow);
    float weightSpatial = 1.0f;



    //---------------------------------------------------------------------------------------------
    // calculate result
    //---------------------------------------------------------------------------------------------
    float likelihood = weightSpatial * spatialLogLikelihood + weightColor * colorLogLikelihood + weightFlow * flowLogLikelihood;
    ((float*)((char*)outptr__likelihoods + y*step))[x] = likelihood;

}

void SegmentationKernel::calculateLikelihood(cv::gpu::GpuMat &in__colorLogLikelihoods, cv::gpu::GpuMat &in__flowLogLikelihoods, cv::gpu::GpuMat &in__sumOfSpatialMeans, cv::gpu::GpuMat &in__maxFlowLogLikelihoods, int numberOfDataPoints, float sigma, int halfSearchRegion, cv::gpu::GpuMat &out__likelihoods) {

    float pi = 3.14159265358979323846f;
    int size = in__colorLogLikelihoods.cols * in__colorLogLikelihoods.cols;
    int threadSize = 1024;
    int blockSize = (size / threadSize) + 1;

    dv__calculateLikelihood<<<blockSize, threadSize>>>(
                            in__colorLogLikelihoods.ptr<float>(), in__flowLogLikelihoods.ptr<float>(),
                            in__sumOfSpatialMeans.ptr<float>(), in__maxFlowLogLikelihoods.ptr<float>(),
                            in__colorLogLikelihoods.step,
                            in__colorLogLikelihoods.cols, in__colorLogLikelihoods.rows,
                            numberOfDataPoints,
                            sigma,
                            halfSearchRegion,
                            pi,
                            out__likelihoods.ptr<float>());
}

__global__ void dv__getBiggerValue(float *exptr__maxLikelihood, int *exptr__maxclass, float *inptr__likelihood, int classK, size_t step, int cols, int rows) {

    //---------------------------------------------------------------------------------------------
    // setup variables
    //---------------------------------------------------------------------------------------------
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int size = cols * rows;
    int x = idx % cols;
    int y = idx / cols;



    //---------------------------------------------------------------------------------------------
    // bounds check
    //---------------------------------------------------------------------------------------------
    if (idx >= size) return;



    //---------------------------------------------------------------------------------------------
    // get bigger likelihood
    //---------------------------------------------------------------------------------------------
    float likelihood = ((float*)((char*)inptr__likelihood + y*step))[x];
    float maxlikelihood = ((float*)((char*)exptr__maxLikelihood + y*step))[x];
    if (likelihood > maxlikelihood) {
        ((float*)((char*)exptr__maxLikelihood + y*step))[x] = likelihood;
        ((int*)((char*)exptr__maxclass + y*step))[x] = classK;
    }

}

void SegmentationKernel::getBiggestLikelihood(cv::gpu::GpuMat &ex__maxLikelihood, cv::gpu::GpuMat &ex__maxclass, cv::gpu::GpuMat &in__likelihood, int classK) {

    int size = ex__maxLikelihood.cols * ex__maxLikelihood.rows;
    int threadSize = 1024;
    int blockSize = (size / threadSize) + 1;

    dv__getBiggerValue<<<blockSize, threadSize>>>(ex__maxLikelihood.ptr<float>(), ex__maxclass.ptr<int>(), in__likelihood.ptr<float>(), classK, ex__maxLikelihood.step, ex__maxLikelihood.cols, ex__maxLikelihood.rows);

}

