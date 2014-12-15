#include <iostream>
#include <cstdio>
#include <Eigen/Dense>
#include "opencv2/opencv.hpp"

// CUDA runtime.
#include <cuda.h>
#include <cuda_runtime.h>
//ADD_VECT library header
#include "add_vect.h"
//add opencv
#include <opencv2/gpu/gpu.hpp>


#define SIZE 1024


using namespace Eigen;
using namespace cv;
using namespace std;



void testOpenCVAndCuda() {
    // Load images
    cv::Mat PreviousFrameGrayFloat = Mat::eye(60, 60, CV_32FC1); // Has an image in format CV_32FC1
    cv::Mat CurrentFrameGrayFloat = Mat::eye(60, 60, CV_32FC1);  // Has an image in format CV_32FC1

    // Upload images to GPU
    cv::gpu::GpuMat PreviousFrameGPU(PreviousFrameGrayFloat);
    cv::gpu::GpuMat CurrentFrameGPU(CurrentFrameGrayFloat);

    // Prepare receiving variables
    cv::gpu::GpuMat FlowXGPU;
    cv::gpu::GpuMat FlowYGPU;

    // Create optical flow object
    cv::gpu::BroxOpticalFlow OpticalFlowGPU = cv::gpu::BroxOpticalFlow(0.197f, 0.8f, 50.0f, 10, 77, 10);

    // Perform optical flow
    OpticalFlowGPU(PreviousFrameGPU, CurrentFrameGPU, FlowXGPU, FlowYGPU); // EXCEPTION
    // Exception in opencv_core244d!cv::GlBuffer::unbind

    // Download flow from GPU
    cv::Mat FlowX;
    cv::Mat FlowY;
    FlowXGPU.download(FlowX);
    FlowYGPU.download(FlowY);
}

void testCuda() {
    //vectors
    const int N = 5;

    double A[N] = {31.23, 321.45, 431.123, 98, 762.14};
    double B[N] = {12.3, 3.2145, 432.3, 982.3, 7621.4};
    double C[N] = {0.0, 0.0, 0.0, 0., 0.0};


    //GPU variables
    double *dev_A, *dev_B, *dev_C;
    cudaMalloc(&dev_A, N * sizeof(double));
    cudaMalloc(&dev_B, N * sizeof(double));
    cudaMalloc(&dev_C, N * sizeof(double));
    cudaMemcpy(dev_A, A, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, C, N * sizeof(double), cudaMemcpyHostToDevice);

    //call the library function
     add_vect(N, dev_A, dev_B, dev_C);

    //copy back
    cudaMemcpy(C, dev_C, N * sizeof(double), cudaMemcpyDeviceToHost);

    // print results
    for (int i=0; i<N; i++) {
        printf("elem %d-th = %f\n",i,C[i]);
    }
}

int testOpencv() {
        VideoCapture cap(0); // open the default camera
        if(!cap.isOpened())  // check if we succeeded
            return -1;
        Mat frame;

        // ensure that videocapture is initialized
        while(frame.rows == 0 && frame.cols == 0) {
            cap >> frame;
        }

        // create window with dimensions of frame
        namedWindow("edges",1);

        // capture frame
        while(1) {
            cap >> frame;
            imshow("edges", frame);
            if(waitKey(30) >= 0) break;
        }

        // the camera will be deinitialized automatically in VideoCapture destructor
        return 0;
}

void testEigen() {
    Matrix3d m = Matrix3d::Random();
    m = (m + Matrix3d::Constant(1.2)) * 50;
    cout << "m =" << endl << m << endl;
    Vector3d v(1,2,3);
    cout << "m * v =" << endl << m * v << endl;
}
