// ============================================================================
//  
//  Project 2.2: 3D Object Tracking (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
//  
//  Source:      https://github.com/udacity/SFND_3D_Object_Tracking
//
// ============================================================================

// function declarations for Lidar point cloud data processing and visualization

#ifndef lidarData_hpp
#define lidarData_hpp

#include <stdio.h>
#include <fstream>
#include <string>

#include "dataStructures.h"

void cropLidarPoints(
    std::vector<LidarPoint> &lidarPoints, float minX, float maxX, float maxY,
    float minZ, float maxZ, float minR);

void loadLidarFromFile(
    std::vector<LidarPoint> &lidarPoints, std::string filename);

void showLidarTopview(
    std::vector<LidarPoint> &lidarPoints, cv::Size worldSize, cv::Size imageSize,
    bool bWait = true, std::string windowName = "Top-View Perspective of LiDAR data");

void showLidarImgOverlay(
    cv::Mat &img, std::vector<LidarPoint> &lidarPoints, cv::Mat &P_rect_xx,
    cv::Mat &R_rect_xx, cv::Mat &RT, cv::Mat *extVisImg=nullptr, bool bWait = true,
    std::string windowName = "Top-View Perspective of LiDAR data on image overlay");
    
#endif /* lidarData_hpp */
