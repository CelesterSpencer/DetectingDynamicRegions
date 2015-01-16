#include "SegmentationKernel.h"

//---------------------------------------------------------------------------------------------------------------------------------
// GLOBAL MOTION ESTIMATION
//---------------------------------------------------------------------------------------------------------------------------------

__global__ void fillBinsDevice(float *flowAngle, size_t stepAngle, float *flowMagnitude, size_t stepMagnitude, int cols, int rows, int numberOfAngles, int numberOfMagnitudes, float legthPerMagnitude, int *md__bins) {

   int size = rows * cols;
   int numberOfBins = numberOfAngles * numberOfMagnitudes;

   // get position within opticalflowfield
   int idx = threadIdx.x + blockIdx.x * blockDim.x;
   int y = idx / cols;
   int x = idx % cols;

   //bounds checking
   if (idx >= size) return;
   if (x >= cols || y >= rows) return;

   // get angle for calculated position
   float angle = ((float*)((char*)flowAngle + y*stepAngle))[x];
   //printf("angle: %f \n", angle);

   // calculate angleNumber of current thread
   int degreePerAngle = 360 / numberOfAngles;
   int angleNumber = angle / degreePerAngle;
   if (angleNumber == 16) angleNumber = 0; // 360° == 0°

   // bounds check
   if (angleNumber >= numberOfAngles)  {
       printf("Angle is bigger than expected: \n");
       printf("Idx: %d, y: %d, x: %d, angle: %f, Binnumber: %d \n", idx, y, x, angle, angleNumber);
       return;
   }

   // get magnitude for calculated position
   float magnitude = ((float*)((char*)flowMagnitude + y*stepMagnitude))[x];
   //printf("magnitude: %f \n", magnitude);

   int magnitudeNumber = magnitude / legthPerMagnitude;

   // bounds check
   if (magnitudeNumber >= numberOfMagnitudes)  {
       printf("magnitude is bigger than expected: \n");
       printf("Idx: %d, y: %d, x: %d, magnitude: %f, Magnitudenumber: %d \n", idx, y, x, magnitude, magnitudeNumber);
       return;
   }

   // increase bin
   md__bins[idx * numberOfBins + angleNumber * numberOfMagnitudes + magnitudeNumber]++;
}

__global__ void sumUpAllBins(int numberOfPixels, bool isOdd, int numberOfBins, int *md__bins) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int binNumber = idx % numberOfBins;
   int pixelNumber = idx / numberOfBins;
   int halfSize = (numberOfPixels / 2);

   // bound check
   if (pixelNumber < halfSize) {

       //add the values of two bins together
       int idx2 = (pixelNumber+halfSize)*numberOfBins + binNumber;
       if (isOdd && (pixelNumber == halfSize - 1)) {
           int idx3 = idx2 + numberOfBins;
           int result = md__bins[idx] + md__bins[idx2] + md__bins[(idx3)];
           md__bins[idx] = result;
       }else {
           int result = md__bins[idx] + md__bins[idx2];
           md__bins[idx] = result;
       }
   }
}

__host__ void SegmentationKernel::fillBins(float* inptr__flowVector3DMagnitude, float* inptr__flowVector3DAngle, size_t in__flowVector3DMagnitudeStep, size_t in__flowVector3DAngleStep,
              int in__cols, int in__rows, int in__numberOfMagnitudes, int in__numberOfAngles, float in__lengthPerMagnitude, int* outptr__bins) {
    int size = in__cols * in__rows;
    int blockSize = ceil((float)size/m__threadSize);
//    printf("Block dimension: %d Thread dimension %d \n", blockSize, m__threadSize);
    fillBinsDevice<<<blockSize, m__threadSize>>>(
        inptr__flowVector3DAngle,
        in__flowVector3DAngleStep,
        inptr__flowVector3DMagnitude,
        in__flowVector3DMagnitudeStep,
        in__cols,
        in__rows,
        in__numberOfAngles,
        in__numberOfMagnitudes,
        in__lengthPerMagnitude,
        outptr__bins
    );
    cudaDeviceSynchronize();
}

__host__ void SegmentationKernel::sumUpBins(int in__tempSize, bool in__isOdd, int in__numberOfBins, int* outptr__bins) {
    int blockSize = ceil((float)(in__tempSize * in__numberOfBins) / m__threadSize);
    if (blockSize <= 0) blockSize = 1;
//    printf("Block dimension: %d Thread dimension %d \n", blockSize, m__threadSize);
    sumUpAllBins<<<blockSize, m__threadSize>>>(in__tempSize, in__isOdd, in__numberOfBins, outptr__bins);
    cudaDeviceSynchronize();
}

//---------------------------------------------------------------------------------------------------------------------------------
// GLOBAL MOTION SUBTRACTION
//---------------------------------------------------------------------------------------------------------------------------------

__global__ void globalMotionSubtractionDevice(float *flowMatX, float *flowMatY, int rows, int cols, size_t step, float globalX, float globalY, float *flowXSubtracted, float *flowYSubtracted) {

    // values
    int size = rows * cols;

    // get position within opticalflowfield
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y = idx / cols;
    int x = idx % cols;

    // bounds checking
    if (idx < size) {
        float xElement = ((float*)((char*)flowMatX + y*step))[x];
        float yElement = ((float*)((char*)flowMatY + y*step))[x];

        ((float*)((char*)flowXSubtracted + y*step))[x] = xElement - globalX;
        ((float*)((char*)flowYSubtracted + y*step))[x] = yElement - globalY;
    }

}

__host__ void SegmentationKernel::globalMotionSubtractionHost(cv::gpu::GpuMat &flow3DX, cv::gpu::GpuMat &flow3DY, float globalX, float globalY, cv::gpu::GpuMat flowXSubtracted, cv::gpu::GpuMat flowYSubtracted) {

    int size = flow3DX.rows * flow3DX.cols;

    // iterate over frame and subtract x and y
    int blockSize = ceil((float)size / m__threadSize);
    if (blockSize <= 0) blockSize = 1;
//    printf("Block dimension: %d Thread dimension %d \n", blockSize, m__threadSize);
    globalMotionSubtractionDevice<<<blockSize, m__threadSize>>>(flow3DX.ptr<float>(), flow3DY.ptr<float>(), flow3DX.rows, flow3DX.cols, flow3DX.step, globalX, globalY, flowXSubtracted.ptr<float>(), flowYSubtracted.ptr<float>());
    cudaDeviceSynchronize();
}



//---------------------------------------------------------------------------------------------------------------------------------
// Segmentation
//---------------------------------------------------------------------------------------------------------------------------------

