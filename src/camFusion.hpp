// ============================================================================
//  
//  Project 2.2: 3D Object Tracking (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
//  
//  Source:      https://github.com/udacity/SFND_3D_Object_Tracking
//
// ============================================================================

// function declarations for camera image and Lidar point cloud data fusion

#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>
#include <vector>
#include <stdexcept>
#include <opencv2/core.hpp>
#include "dataStructures.h"

void clusterLidarWithROI(
    std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints,
    float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT);

void clusterKptMatchesWithROI(
    BoundingBox &prevBB, BoundingBox &currBB, std::vector<cv::KeyPoint> &kptsPrev,
    std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches);

void matchBoundingBoxes(
    std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches,
    DataFrame &prevFrame, DataFrame &currFrame);

void show3DObjects(
    std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize,
    cv::Size imageSize, bool bWait = true, std::string windowName = "3D Objects");

void computeTTCCamera(
    BoundingBox &prevBB, BoundingBox &currBB,
    std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
    std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC,
    cv::Mat *visImg = nullptr, bool bVis = false, bool bWait = true,
    bool bPrintDebugInfo = false);
    
void computeTTCLidar(
    std::vector<LidarPoint> &lidarPointsPrev, std::vector<LidarPoint> &lidarPointsCurr,
    double frameRate, double &TTC, int option,
    BoundingBox &currBB, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT,
    cv::Mat *visImg = nullptr, bool bVis = false, bool bWait = true);

// calculate the mean of a vector
template <class T>
T mean(std::vector<T> vec)
{
    // get vector size
    auto n = vec.size();
    if (n == 0)
        throw std::domain_error("mean of an empty vector");
    
    // calculate mean value
    T cumsum = 0.0;
    for ( auto i = 0; i < n; i++ )
    {
        cumsum += vec[i];
    }
    return cumsum/n;
}

// calculate the median of a vector
template <class T>
T median(std::vector<T> vec)
{
    // get vector size
    auto n = vec.size();
    if (n == 0)
        throw std::domain_error("median of an empty vector");
    
    // sort the elements of the vector in non-descending order
    sort(vec.begin(), vec.end());

    // get median value
    auto mid = n/2;
    return n % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
}

// calculate mean and standard deviation of a vector
template <class T>
std::pair<T, T> calcMeanAndStandardDeviation(std::vector<T> vec)
{
    // get vector size
    auto n = vec.size();
    if (n == 0)
        throw std::domain_error("standard deviation of an empty vector");
    
    // declar output argument
    std::pair<T, T> out;

    // calculate mean value
    T sum = 0.0;
    for ( auto i = 0; i < n; i++ )
    {
        sum += vec[i];
    }
    T mean_value = sum/n;

    // calculate standard deviation
    T squared_sum = 0.0;
    for ( auto i = 0; i < n; i++ )
    {
        squared_sum += (vec[i] - mean_value) * (vec[i] - mean_value);
    }
    T standard_deviation = sqrt(squared_sum/n);
    
    // return mean value and standard deviation as pair of values
    return std::make_pair(mean_value, standard_deviation);
}

#endif /* camFusion_hpp */
