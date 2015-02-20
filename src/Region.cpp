#include "Region.h"

Region::Region(int cols, int rows) {
    m__minX = cols;
    m__minY = rows;
}

void Region::addPixel(int x, int y) {
    pixels.push_back(cv::Point(x,y));
    if (x < m__minX) m__minX = x;
    if (y < m__minY) m__minY = y;
    if (x > m__maxX) m__maxX = x;
    if (y > m__maxY) m__maxY = y;
}

std::vector<cv::Point> Region::getAllPixels() {
    return pixels;
}
