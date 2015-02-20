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
    Region(int cols, int rows);
    void addPixel(int x, int y);
    std::vector<cv::Point> getAllPixels();
private:
    std::vector<cv::Point> pixels;
    int m__minX = -1;
    int m__maxX = -1;
    int m__minY = -1;
    int m__maxY = -1;
    int m__centerX = -1;
    int m__centerY = -1;
    float m__varianceX = -1.0;
    float m__varianceY = -1.0;
};

#endif
