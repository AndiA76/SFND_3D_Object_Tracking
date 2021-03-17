// ============================================================================
//  
//  Project 2.2: 3D Object Tracking (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
//  
//  Source:      https://github.com/udacity/SFND_3D_Object_Tracking
//
// ============================================================================

// function declarations for 2D keypoint detection & 2D feature matching

#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

void limitKeypoints(
    std::vector<cv::KeyPoint> & keypoints, cv::Mat & imgGray, std::string detectorType,
    int maxKpts, bool bVis, bool bWait);

bool compareKeypointResponse(
    const cv::KeyPoint & p1, const cv::KeyPoint & p2);

double detKeypointsShiTomasi(
    std::vector<cv::KeyPoint> & keypoints, cv::Mat & img,
    bool bVis = false, bool bWait = true);

double detKeypointsHarris(
    std::vector<cv::KeyPoint> & keypoints, cv::Mat & img,
    bool bVis = false, bool bWait = true);

double detKeypointsModern(
    std::vector<cv::KeyPoint> & keypoints, cv::Mat & img, std::string detectorType,
    bool bVis = false, bool bWait = true);

double descKeypoints(
    std::vector<cv::KeyPoint> & keypoints, cv::Mat & img, cv::Mat & descriptors,
    std::string descExtractorType);

double matchDescriptors(
    std::vector<cv::KeyPoint> & kPtsSource, std::vector<cv::KeyPoint> & kPtsRef,
    cv::Mat & descSource, cv::Mat & descRef, std::vector<cv::DMatch> & kPtMatches,
    std::string descriptorType, std::string matcherType, std::string selectorType);

void showKptMatches(
    std::vector<cv::KeyPoint> & kPtsSource, std::vector<cv::KeyPoint> & kPtsRef, 
    cv::Mat & kPtsSourceImage, cv::Mat & kPtsRefImage, std::vector<cv::DMatch> & kPtMatches,
    bool bWait = true, std::string windowName = "Matching keypoints between two camera images");

void showKptMatchesWithROI(
    std::vector<cv::KeyPoint> & kPtsSource, std::vector<cv::KeyPoint> & kPtsRef, 
    cv::Mat & kPtsSourceImage, cv::Mat & kPtsRefImage, std::vector<cv::DMatch> & kPtMatches,
    cv::Rect & kPtsSourceROI, cv::Rect & kPtsRefROI, 
    bool bWait = true, std::string windowName = "Matching keypoints between two camera images");

#endif /* matching2D_hpp */
