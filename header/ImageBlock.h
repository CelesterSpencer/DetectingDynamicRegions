#ifndef _IMAGE_BLOCK_H_
#define _IMAGE_BLOCK_H_

#include <opencv2/core/core.hpp>
#include "Region.h"

enum Distance {
    FAR,
    MIDDLE,
    NEAR,
};



/*!
 * \class A part of an frame that is used to break down and parallelize computation
 * \author Adrian Derstroff
 * \date 09.12.2014
 */
class ImageBlock{
public:
    float m__clarity;
    float m__contrast;
    Distance m__distanceLevel;
    Distance m__distanceLevelContrast;
    Distance m__distanceCombined;
    int startX;
    int endX;
    int startY;
    int endY;
#endif
