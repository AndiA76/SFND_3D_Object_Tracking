// ============================================================================
//  
//  Project 2.2: 3D Object Tracking (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
//  
//  Source:      https://github.com/udacity/SFND_3D_Object_Tracking
//
// ============================================================================

// function declarations for 2D object detection in color images

#ifndef objectDetection2D_hpp
#define objectDetection2D_hpp

#include <stdio.h>
#include <opencv2/core.hpp>

#include "dataStructures.h"

void detectObjects(
    cv::Mat& img, std::vector<BoundingBox>& bBoxes, float confThreshold, float nmsThreshold,
    std::string basePath, std::string classesFile, std::string modelConfiguration,
    std::string modelWeights, bool bVis, bool bWait);

#endif /* objectDetection2D_hpp */
