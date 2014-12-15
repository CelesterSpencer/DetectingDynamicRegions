#ifndef _MOTION_HISTORY_IMAGE_H_
#define _MOTION_HISTORY_IMAGE_H_

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <time.h>
#include <stdio.h>
#include <ctype.h>

using namespace cv;

class MotionHistoryImage {
public:
	// parameters:
	//  img - input video frame
	//  dst - resultant motion picture
	//  args - optional parameters
    void  update_mhi( Mat* img, Mat* dst, int diff_threshold );

private:

};

#endif
