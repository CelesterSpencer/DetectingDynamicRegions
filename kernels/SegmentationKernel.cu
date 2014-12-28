 #include "SegmentationKernel.h"


__global__ void getGlobalMotionDevice(float *flowAngle, size_t step, int cols, int rows, float *flowMagnitude, float *globalMotion, int numberOfBins, int *md__bins) {

    int size = rows * cols;

    // get position within opticalflowfield
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y = idx / cols;
    int x = idx % cols;
//    printf("Idx: %d x: %d, y: %d \n", idx, x, y);

    //bounds checking
    if (idx >= size) return;
    if (x >= cols || y >= rows) return;

    // get value for calculated position
    float angle = ((float*)((char*)flowAngle + y*step))[x];      //STEP MIGHT BE WRONG IN THIS CONTEXT

    // calculate binNumber of current thread
    int degreePerBin = 360 / numberOfBins;
    int binNumber = angle / degreePerBin;

//    ((float*)((char*)flowAngle + y*step))[x] = 0;
    printf("numOfBins: %d, binNumber: %d, angle: %f, idx: %d \n", numberOfBins, binNumber, angle, idx);

    // bounds check
    if (binNumber >= numberOfBins)  {
        printf("Bin is bigger than expected: %d \n", binNumber);
        return;
    }

    // calculate global motion
    md__bins[idx * numberOfBins + binNumber]++;
}

//TODO HERE ARE STILL SOME ERRORS
__global__ void sumUpAllBins(int numberOfThreads, bool isOdd, int numberOfBins, int *md__bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numberOfThreads) {

        //add the values of two bins together
        int idx2 = idx + numberOfThreads;
        if (isOdd && idx == (numberOfThreads - 1)) {
            for(int ind = 0; ind < numberOfBins; ind++) {
                int value1 = md__bins[idx2 * numberOfBins + ind];
                int value2 = md__bins[(idx2+1) * numberOfBins + ind];
                int oldValue = md__bins[idx * numberOfBins + ind];
                int result = oldValue + value1 + value2;
                md__bins[idx * numberOfBins + ind] = result;
            }
        }else {
            for(int ind = 0; ind < numberOfBins; ind++) {
                int value = md__bins[idx2 * numberOfBins + ind];
                int oldValue = md__bins[idx * numberOfBins + ind];
                int result = oldValue + value;
                md__bins[idx * numberOfBins + ind] = result;

            }
        }
    }
}

__host__ void SegmentationKernel::getGlobalMotionHost(cv::gpu::GpuMat &flow3DAngle, cv::gpu::GpuMat &flow3DMagnitude, int numberOfBins, int threadsize) {

    // calculate blocksize
    int size = flow3DAngle.rows * flow3DAngle.cols;
    int cols = flow3DAngle.cols;
    int rows = flow3DAngle.rows;

    // allocate array on device
    int *d__bins;
    int *binsPtr = new int[size * numberOfBins];
    for(int ind = 0; ind < size * numberOfBins; ind++) {
        *(binsPtr + ind) = 0;
    }
    cudaMalloc((void **)&d__bins, sizeof(int) * size * numberOfBins);
    cudaMemcpy(d__bins, binsPtr, sizeof(int) * size * numberOfBins, cudaMemcpyHostToDevice);

    // copy GpuMat data to device
    float *dataMagnitude = (float*)flow3DMagnitude.data;
    float *dataGlobalMotion = new float[2];
    float *d__magnitude;
    float *d__globalMotion;
    cudaMalloc((void**)&d__magnitude, sizeof(float) * size);
    cudaMalloc((void**)&d__globalMotion, sizeof(float) * 2); // global motion is a vector (angle, magnitude)^T
    cudaMemcpy(d__magnitude, dataMagnitude, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d__globalMotion, dataGlobalMotion, sizeof(float) * 2, cudaMemcpyHostToDevice);

    //-------------------------------------------------------------------------------
//    cv::Mat mat = cv::Mat(10, 10, CV_32FC1);
//    float value = 0.0;
//    for (int y = 0; y < mat.cols; y++) {
//        for (int x = 0; x < mat.rows; x++) {
//            mat.at<float>(y,x) = value;
//            value += 3;
//        }
//    }
//    cv::gpu::GpuMat testMat = cv::gpu::GpuMat(10, 10, CV_32FC1);
//    testMat.upload(mat);
//    int testRows = testMat.rows;
//    int testCols = testMat.cols;
//    int testSize = testRows * testCols;
//    // allocate array on device
//    int *testbins;
//    int *testbinsPtr = new int[testSize * numberOfBins];
//    for(int ind = 0; ind < testSize * numberOfBins; ind++) {
//        *(testbinsPtr + ind) = 0;
//    }
//    cudaMalloc((void **)&testbins, sizeof(int) * testSize * numberOfBins);
//    cudaMemcpy(testbins, testbinsPtr, sizeof(int) * testSize * numberOfBins, cudaMemcpyHostToDevice);
    //-------------------------------------------------------------------------------

    // calculate corresponding bin for every entry in d__angle
    int threadSize = 64;
    int blockSize = ceil((float)size/threadSize);
    printf("Block dimension: %d Thread dimension %d \n", blockSize, threadSize);
    getGlobalMotionDevice<<<blockSize, threadSize>>>(flow3DAngle.ptr<float>(), flow3DAngle.step, cols, rows, d__magnitude, d__globalMotion, numberOfBins, d__bins);
    cudaDeviceSynchronize();

    // collect results iteratively
    int tempSize = size;
    while(1) {
        int numberOfThreads = tempSize / (2);
        if (numberOfThreads >= 1) {
            int blocksize = tempSize / threadsize;
            if (blocksize <= 0) blocksize = 1;
            bool isOdd = (numberOfThreads*2 != tempSize);
            sumUpAllBins<<<blocksize, threadsize>>>(numberOfThreads, isOdd, numberOfBins, d__bins);
            cudaDeviceSynchronize();
            tempSize /= 2;
        }else {
            break;
        }
    }

    // data must eventually be changed
    cudaMemcpy(dataGlobalMotion, d__globalMotion, sizeof(float) * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(binsPtr, d__bins, sizeof(int) * size * numberOfBins, cudaMemcpyDeviceToHost);

    // sum up remaining bins to check wether their sum equals the number of pixel of the frame
    printf("Bins: \n");
    int resultNumberOfPixel = 0;
    for(int ind = 0; ind < numberOfBins; ind++) {
        int value = *(binsPtr + ind);
        resultNumberOfPixel += value;
        printf("Bin %d : %d \n", ind, value);
    }
    if (resultNumberOfPixel == size) {
        printf("calculated value and size are equal! \n");
    } else {
        printf("calculated value and size are not equal! %d and %d \n", resultNumberOfPixel, size);
    }

    free(binsPtr);
    free(dataGlobalMotion);
    cudaFree(d__magnitude);
    cudaFree(d__globalMotion);
}

