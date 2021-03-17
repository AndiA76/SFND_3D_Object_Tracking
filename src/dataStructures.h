// ============================================================================
//  
//  Project 2.2: 3D Object Tracking (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
//  
//  Source:      https://github.com/udacity/SFND_3D_Object_Tracking
//
// ============================================================================

// image frame, 3D Lidar point cloud and evaluation result data structure definitions

#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <map>
#include <opencv2/core.hpp>

struct LidarPoint { // single 3D lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
};

struct BoundingBox { // bounding box around a classified object (contains both 2D and 3D data)
    
    // object bounding box and track ids
    int boxID; // unique identifier for this bounding box
    int trackID; // unique identifier for the track to which this bounding box belongs
    
    // object detections
    cv::Rect roi; // 2D region-of-interest in image coordinates
    int classID; // ID based on class file provided to YOLO framework
    double confidence; // classification trust

    // 3D Lidar points and 2D keypoint matches
    std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image ROI
    std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D ROI
    std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D ROI
};

struct DataFrame { // represents the available sensor information at the same time instance
    
    // raw color image data
    std::string imgFilename; // camera image file name
    cv::Mat cameraImg; // camera image (color information is needed for CNN-based object detection)
    cv::Mat cameraImgGray; // camera image converted to grayscale

    // keypoints and keypoint descriptors
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors for each keypoint within camera image
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame

    // 3D Lidar points
    std::vector<LidarPoint> lidarPoints;

    // object bounding boxes
    std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
    std::map<int,int> bbMatches; // bounding box matches between previous and current frame
};

struct EvalResults { // represents the evaluation results collected over a variable number of data frames

    // image name
    std::string imgFilename; // camera image file name

    // configuration parameters
    std::string detectorType; // keypoint destector type
    bool bLimitKpts; // force limitation of detected keypoints => only for debugging => should be false
    std::string descExtractorType; // keypoint descriptor extractor type
    std::string matcherType; // descriptor matcher type
    std::string descriptorType; // descriptor type
    std::string selectorType; // selector type

    // evaluation results
    int numKeypoints; // number of keypoints found in the image
    int numKeypointsLimited; // limited number of detected keypoints found in the image
    int numDescMatches; // number of matched keypoints within the region of interest
    double meanDetectorResponse; // mean keypont detector response
    double meanKeypointDiam; // mean keypoint diameter
    double varianceKeypointDiam; // variance of keypoint diameters
    double t_detKeypoints; // processing time needed for keypoint detection (all keypoints)
    double t_descKeypoints; // processing time needed for keypoint descriptor extraction (keypoints in ROI)
    double t_matchDescriptors; // processing time needed for keypoint descriptor matching (keypoints in ROI)
    double t_sum_det_desc; // t_detKeypoints + t_descKeypoints
    double t_sum_det_desc_match; // t_detKeypoints + t_descKeypoints + t_matchDescriptors
    int numLidarPointsOnTarget; // number of Lidar points associated with the target object
    double ttcLidar; // time-to-collision estimate using Lidar data
    int numKptMatchesOnTarget; // number of keypoint matches associated with the target object
    double ttcCamera; // time-to-collision estimate using camera data
};

#endif /* dataStructures_h */
