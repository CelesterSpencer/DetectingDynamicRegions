#ifndef _REGION_H_
#define _REGION_H_

#include <opencv2/core/core.hpp>

/*!
 * \class A set of all pixel within a frame that belong to the same object
 * \author Adrian Derstroff
 * \date 09.12.2014
 */
class Region {
public:

private:
    cv::Rect m__boundingBox; //enclosing possible area of pixel that belong to this region
    cv::Mat m__mask;
};

#endif
