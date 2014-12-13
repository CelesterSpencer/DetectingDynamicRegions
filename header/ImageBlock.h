#ifndef _IMAGE_BLOCK_H_
#define _IMAGE_BLOCK_H_

#include <opencv2/core/core.hpp>
#include "Region.h"

/*!
 * \class A part of an frame that is used to break down and parallelize computation
 * \author Adrian Derstroff
 * \date 09.12.2014
 */
class ImageBlock{
public:
    /*!
     * \brief an image block's can be in the foreground which is near, in the middle or in the background which is far
     */
    enum Distance {NEAR, MIDDLE, FAR};
    /*!
     * \brief calculate depth and label it with either near, middle or far
     */
    void calDistance();
private:
    cv::Mat m__inputFrame;
    Region m__region;
    Distance m__distanceLabel;
    double m__distanceValue;
};

#endif
