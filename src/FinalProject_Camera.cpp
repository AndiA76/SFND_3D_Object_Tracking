
// ============================================================================
//  
//  Project 2.2: 3D Object Tracking (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
//  
//  Source:      https://github.com/udacity/SFND_3D_Object_Tracking
//
// ============================================================================

// main functions for 3D object tracking fusing vision and Lidar data

/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
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

#include <boost/circular_buffer.hpp>
#include <boost/circular_buffer/base.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"
#include "exportResults.hpp"

using namespace std;


/* GLOBAL VARIABLES */

// define positive and negative infinity
double POS_INF = 1.0 /0.0;
double NEG_INF = -1.0/0.0;

// root directory (and data location)
string rootDir = "../";

// result base path
string resultBasePath = rootDir + "results/";


/* SUBFUNCTIONS */

// 3D Object tracking function using fusion of 3D Lidar and 2D vision data
int track3DObjects(
    vector<boost::circular_buffer<EvalResults>> & evalResultBuffers,
    std::string detectorType = "FAST",
    bool bLimitKpts = false,
    int maxKpts = 100,
    std::string descExtractorType = "BRIEF",
    std::string matcherType = "MAT_BF",
    std::string descriptorType = "DES_BINARY",
    std::string selectorType = "SEL_KNN",
    int TTCLidarEstimationOption = 4,
    bool bVis = true,
    bool bVisDebug = true,
    bool bPrintDebugInfo = false,
    bool bWait = true,
    bool bSaveImageToFile = false,
    bool bExportResultsToCSV = true)
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // camera (vision)
    string imgBasePath = rootDir + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection CNN (vision)
    string yoloBasePath = rootDir + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar (needs to be adjusted to the individual measurement set up of the recorded data sequences)
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // intrinsic 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // intrinsic 3x3 rectifying rotation to make left and right stereo image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // extrinsic rotation matrix and translation vector of the camera mounting position
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // sensor frame rates
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera

    /* using boost circular data buffer class template
    *  https://www.boost.org/doc/libs/1_65_1/doc/html/circular_buffer.html
    */
    // initialze circular data buffer
    int dataBufferSize = 2; // no. of images which are held in memory (ring buffer) at the same time
    boost::circular_buffer<DataFrame> dataBuffer(dataBufferSize); // create circular buffer for DataFrame structures

    // print data buffer capacity
    cout << "Circular data buffer capacity in use = "
        << dataBuffer.size()
        << " of max. "
        << dataBuffer.capacity()
        << " data frames."
        << endl;

    // skip combination flag if keypoint detector - descriptor extractor combination is not compatible
    bool bSkipCombination = false;

    // circular data buffer to hold evaluation results
    int resultBufferSize = imgEndIndex - imgStartIndex + 1;
    boost::circular_buffer<EvalResults> resultBuffer(resultBufferSize);

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {

        /* #1: LOAD COLOR IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load color image from file
        cv::Mat img = cv::imread(imgFullFilename);

        // push color image and file name back to the tail of the circular data frame buffer
        DataFrame frame;
        frame.imgFilename = imgNumber.str() + imgFileType;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);

        // clear temporary variables that are no longer needed (avoid memory leaks)
        img.release();

        // create structure to hold the evaluation results and push it back to tail of the circular result buffer, or ringbuffer, resp.
        EvalResults results;
        results.imgFilename = imgNumber.str() + imgFileType;
        resultBuffer.push_back(results);
        
        // EOF #1: LOAD IMAGE INTO BUFFER
        cout << "#1: LOAD COLOR IMAGE INTO BUFFER done" << endl;

        // Print data buffer capacity in use
        cout << "Circular data buffer capacity in use = "
            << dataBuffer.size()
            << " of max. "
            << dataBuffer.capacity()
            << " data frames."
            << endl;


        /* #2: DETECT & CLASSIFY OBJECTS USING VISION */

        // Set confidence threshold
        float confThreshold = 0.2;

        // Set non-maximum suppression threshold
        float nmsThreshold = 0.4;

        // Detect object in the color image frame stored in the last element of the ringbuffer
        detectObjects(
            (dataBuffer.end()-1)->cameraImg, (dataBuffer.end()-1)->boundingBoxes, confThreshold,
            nmsThreshold, yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis, bWait);

        // EOF #2: DETECT & CLASSIFY OBJECTS USING VISION
        cout << "#2: DETECT & CLASSIFY OBJECTS USING VISION done" << endl;


        /* #3: CROP LIDAR POINTS */
        /*
        *  Please note:
        *  The 3D Lidar points are cropped from the BB ROI of the next target object on ego lane!
        *  All other Lidar points are dropped.
        *  If you want to get the Lidar points on the other target objects, too. this part needs
        *  to be modified. You need to loop over all target objects and crop the Lidar points per
        *  bounding box detection individually.
        */

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties assuming a flat resp. level road surface
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0; // focus on ego lane
        float minR = 0.1; // minimum reflectivity on the road
        // crop only Lidar points within the RO of the next target object in ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
    
        // add Lidar points to the structure of the tail element of the circular data frame buffer
        (dataBuffer.end()-1)->lidarPoints = lidarPoints;

        // clear temporary variables that are no longer needed (avoid memory leaks)
        lidarPoints.clear();

        // Visualize Lidar points from bird's eye perspective (in the first loop only the data from the current frame is available)
        if(bVisDebug)
        {
            // show 3D Lidar points from current frame (top view projection)
            showLidarTopview((dataBuffer.end()-1)->lidarPoints, cv::Size(4.0, 20.0), cv::Size(1000, 1000), bWait);
        }

        // EOF #3: CROP LIDAR POINTS
        cout << "#3: CROP LIDAR POINTS done" << endl;


        /* #4: CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI to prevent Lidar points from other objects close to the current object ROI 
        // creeping into the current object detection ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI(
            (dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end()-1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // Visualize 3D objects (in the first loop only the data from the current frame is available)
        if(bVis)
        {
            // show 3D Lidar objects from current frame (top view projection)
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(1000, 1000), bWait);
        }

        // EOF #4: CLUSTER LIDAR POINT CLOUD
        cout << "#4: CLUSTER LIDAR POINT CLOUD done" << endl;
        

        /* #5: DETECT IMAGE KEYPOINTS */

        // implemented keypoint detector options: "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "KAZE", "AKAZE", "SIFT", "SURF"

        // store selected detector type in result buffer (for all frames)
        (resultBuffer.end() - 1)->detectorType = detectorType;
        cout << "Seletect keypoint detector tpye = " << detectorType << endl;

        // convert current camera image (tail element of the ringbuffer) to grayscale (only for keypoint detection)
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // add grayscale image to the last element of the ringbuffer
        (dataBuffer.end()-1)->cameraImgGray = imgGray;

        // clear temporary variables that are no longer needed (avoid memory leaks)
        imgGray.release();

        // create empty feature list for current image
        vector<cv::KeyPoint> keypoints;

        // Initialize processing time for keypoint detection
        double t_detKeypoints = 0.0;
        
        // detect 2D keypoints in current image
        if (detectorType.compare("SHITOMASI") == 0)
        {
            try
            {
                // Detect keypoints using Shi-Tomasi detector
                t_detKeypoints = detKeypointsShiTomasi(keypoints, (dataBuffer.end()-1)->cameraImgGray, bVisDebug, bWait);
            }
            catch(const exception& e)
            {
                // show exeption and return 1
                cerr << e.what() << endl;
                return 1;
            }
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            try
            {
                // detect keypoints using Harris detector
                t_detKeypoints = detKeypointsHarris(keypoints, (dataBuffer.end()-1)->cameraImgGray, bVisDebug, bWait);
            }
            catch(const exception& e)
            {
                // show exeption and return 1
                cerr << e.what() << endl;
                return 1;
            }
        }
        else
        {
            try
            {
                // detect keypoints using other user-specified detector types
                t_detKeypoints = detKeypointsModern(keypoints, (dataBuffer.end()-1)->cameraImgGray, detectorType, bVisDebug, bWait);
            }
            catch(const char *msg)
            {
                // show error message and return 1
                cout << msg << endl;
                return 1;
            }
            catch(const exception& e)
            {
                // show exeption and return 1
                cerr << e.what() << endl;
                return 1;
            }
        }

        // store the number of detected keypoints and the processing time for keypoint detection in result buffer (for all frames)
        (resultBuffer.end()-1)->numKeypoints = keypoints.size();
        (resultBuffer.end()-1)->t_detKeypoints = t_detKeypoints;

        // optional : limit number of keypoints (helpful only for debugging and learning => Do not use in real application!)
        (resultBuffer.end()-1)->bLimitKpts = bLimitKpts;  // store bLimitKpts flag in result buffer (for all frames)
        cout << "Limit number of keypoints = " << bLimitKpts << endl;
        if (bLimitKpts)
        {
            // limit the number of keypoints (sorted by the strength of the detector response) up to a maximum number
            limitKeypoints(keypoints, (dataBuffer.end()-1)->cameraImgGray, detectorType, maxKpts, bVisDebug, bWait);
        }
        
        // store the number of limited keypoints in result buffer (for all frames)
        (resultBuffer.end()-1)->numKeypointsLimited = keypoints.size(); // equal to number of keypoints in ROI if bLimitKpts == false

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end()-1)->keypoints = keypoints;

        // clear temporary variables that are no longer needed (avoid memory leaks)
        keypoints.clear();

        /* EVALUATE THE MEAN STRENGTH AND NEIGHBORHOOD SIZE (MEAN AND VARIANCE) OF THE REMAINING KEYPOINTS
        *
        *  Note:
        *  - Keypoints.response = strength of the keypoint detectors's response
        *  - keypoints.size = keypoint diameter
        *  - keypoints.size() = length of keypoints vector
        *
        */

        // calculate the mean detector response and the mean keypoint diameter over all remaining keypoints
        double meanDetectorResponse = 0.0;
        double meanKeypointDiam = 0.0;
        for (auto kPt = (dataBuffer.end()-1)->keypoints.begin(); kPt < (dataBuffer.end()-1)->keypoints.end(); kPt++)
        {
            meanDetectorResponse += kPt->response;
            meanKeypointDiam += kPt->size;
        }
        meanDetectorResponse /= (dataBuffer.end()-1)->keypoints.size();
        meanKeypointDiam /= (dataBuffer.end()-1)->keypoints.size();

        // calculate the keypoint diameter variance over all remaining keypoints
        double varianceKeypointDiam = 0.0;
        for (auto kPt = (dataBuffer.end()-1)->keypoints.begin(); kPt < (dataBuffer.end()-1)->keypoints.end(); kPt++)
        {
            varianceKeypointDiam += (kPt->size - meanKeypointDiam) * (kPt->size - meanKeypointDiam);
        }
        varianceKeypointDiam /= (dataBuffer.end()-1)->keypoints.size();

        // output for debugging
        if (bPrintDebugInfo) {
            cout << "Average keypoint detector response:" << endl;
            cout << "meanDetectorResponse = " << meanDetectorResponse << endl;
            cout << "meanKeypointDiam = " << meanKeypointDiam << endl;
            cout << "varianceKeypointDiam = " << varianceKeypointDiam << endl;
        }

        // store the mean strength of the keypoint detector and the statistical neighborhood size in result buffer (for all frames)
        (resultBuffer.end()-1)->meanDetectorResponse = meanDetectorResponse;
        (resultBuffer.end()-1)->meanKeypointDiam = meanKeypointDiam;
        (resultBuffer.end()-1)->varianceKeypointDiam = varianceKeypointDiam;

        // EOF #5: DETECT IMAGE KEYPOINTS
        cout << "#5: DETECT KEYPOINTS done" << endl;


        /* #6: EXTRACT KEYPOINT DESCRIPTORS */

        // implemented keypoint descriptor extractor options: "BRISK", "BRIEF", "ORB", "FREAK", "KAZE", "AKAZE", "SIFT", "SURF"

        // store selected keypoint descriptor type in result buffer (for all frames)
        (resultBuffer.end()-1)->descExtractorType = descExtractorType;
        cout << "Seletect descriptor extractor tpye = " << descExtractorType << endl;

        // initialize descriptor matrix
        cv::Mat descriptors;

        // initialize processing time for keypoint descriptor extraction
        double t_descKeypoints = 0.0;

        // check if descriptor extractor and keypoint detector are compatible
        if ( (detectorType == "SIFT") && (descExtractorType == "ORB") )
        {
            // skip the combination "SIFT" (keypoint detector) and "ORB" (descriptor extractor) due to memory allocation issues
            bSkipCombination = true;

            // inform user about current memory allocation problem using "SIFT" keypoint detector and "ORB"
            cout << "The keypoint detector type "
                << detectorType
                << " in combination with the descriptor extractor type "
                << descExtractorType
                << " causes memory allocation issues ..."
                << endl;
            cout << "... so this combination will be skipped" << endl;
        }
        else if (
            ((descExtractorType != "KAZE") && (descExtractorType != "AKAZE")) ||
            (((descExtractorType == "KAZE") || (descExtractorType == "AKAZE")) && ((detectorType == "KAZE") || (detectorType == "AKAZE")))
            )
        {  // "KAZE" and "AKAZE" descriptor extractors are only compatible with "KAZE" or "AKAZE" keypoints

            // extract keypoint descriptors
            try
            {
                // extract descriptors using user-specified descriptor extractor types
                t_descKeypoints = descKeypoints(
                    (dataBuffer.end()-1)->keypoints, (dataBuffer.end()-1)->cameraImg, descriptors, descExtractorType);
            }
            catch(const char *msg)
            {
                // show error message and return 1
                cout << msg << endl;
                return 1;
            }
            catch(const exception& e)
            {
                // show exeption and return 1
                cerr << e.what() << endl;
                return 1;
            }            
        }
        else
        {
            // skip cominations of "KAZE" and "AKAZE" descriptor extractors with keypoints other than "KAZE" or "AKAZE"
            bSkipCombination = true;

            // inform user about incompatibility of descriptor extractor and keypoint detector type
            cout << "No descriptor extraction possible ... as descriptor extractor type "
                << descExtractorType
                << " is not compatible with keypoint detector type "
                << detectorType
                << endl;            
            cout << "... so this combination will be skipped" << endl;
        }

        // store the processing time for keypoint descriptor extration in result buffer (for all frames)
        (resultBuffer.end()-1)->t_descKeypoints = t_descKeypoints;

        // store the cumulated processing time for keypoint detection and descriptor extraction in result buffer (for all frames)
        (resultBuffer.end()-1)->t_sum_det_desc = t_detKeypoints + t_descKeypoints;

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end()-1)->descriptors = descriptors;

        // clear temporary variables that are no longer needed (avoid memory leaks)
        descriptors.release();

        // EOF #6: EXTRACT KEYPOINT DESCRIPTORS
        cout << "#6: EXTRACT KEYPOINT DESCRIPTORS done" << endl;


        // wait until at least two images have been processed ...
        if (dataBuffer.size() > 1)
        {
            /* #7: MATCH KEYPOINT DESCRIPTORS */

            /* implemented matcher, descriptor and selector options:
            *  matcherType: MAT_BF, MAT_FLANN
            *  descriptorType: DES_BINARY, DES_HOG
            *  selectorType: SEL_NN, SEL_KNN
            */

            // initialize processing time for keypoint descriptor matching
            double t_matchDescriptors = 0.0;

            // check if descriptor extractor and keypoint detector are compatible
            if ( (detectorType == "SIFT") && (descExtractorType == "ORB") )
            { // skip the combination "SIFT" (keypoint detector) and "ORB" (descriptor extractor) due to memory allocation issues

                // inform user about current memory allocation problem using "SIFT" keypoint detector and "ORB"
                cout << "The keypoint detector type "
                    << detectorType
                    << " in combination with the descriptor extractor type "
                    << descExtractorType
                    << " causes memory allocation issues ..."
                    << endl;
                cout << "... so this combination will be skipped" << endl;

                // store configuration for keypoint matching in result buffer (for all frames except for the first frame)
                (resultBuffer.end()-1)->matcherType = matcherType; // store selected matcher type in result buffer
                (resultBuffer.end()-1)->descriptorType = descriptorType; // store selected descriptor type in result buffer
                (resultBuffer.end()-1)->selectorType = selectorType; // store selected selector type in result buffer
                (resultBuffer.end()-1)->numDescMatches = 0; // store number of descriptor matches in result buffer
            }
            else if (
                ((descExtractorType != "KAZE") && (descExtractorType != "AKAZE")) ||
                (((descExtractorType == "KAZE") || (descExtractorType == "AKAZE")) && ((detectorType == "KAZE") || (detectorType == "AKAZE")))
                )
            { // "KAZE" and "AKAZE" descriptor extractors are only compatible with "KAZE" or "AKAZE" keypoints

                // create vector of keypoint descriptor matches
                vector<cv::DMatch> matches;

                // print out configuration for keypoint matching
                cout << "Seletect descriptor matcher tpye = " << matcherType << endl;
                cout << "Seletect descriptor tpye = " << descriptorType << endl;
                cout << "Seletect selector tpye = " << selectorType << endl;

                // print out which images are used
                if (bPrintDebugInfo)
                {
                    cout << "Matching keypoint descriptors between the last and the second last image stored in the ringbuffer:" << endl;
                    cout << "Filename of last image in ringbuffer     = " << (dataBuffer.end()-1)->imgFilename << endl;
                    cout << "Filename of 2nd last image in ringbuffer = " << (dataBuffer.end()-2)->imgFilename << endl;
                }

                try
                {
                    /* match keypoint descriptors between the last and the second last image stored in the ringbuffer, where
                    *  - previous image = source image or query image
                    *  - current image = reference image or train image
                    */
                    t_matchDescriptors = matchDescriptors(
                        (dataBuffer.end()-2)->keypoints, (dataBuffer.end()-1)->keypoints,
                        (dataBuffer.end()-2)->descriptors, (dataBuffer.end()-1)->descriptors,
                        matches, descriptorType, matcherType, selectorType);
                }
                catch(const exception& e)
                {
                    // show exeption and return 1
                    cerr << e.what() << endl;
                    return 1;
                }

                // store matches in current data frame
                (dataBuffer.end()-1)->kptMatches = matches;

                // store configuration for keypoint matching in result buffer (for all frames except for the first frame)
                (resultBuffer.end()-1)->matcherType = matcherType;
                (resultBuffer.end()-1)->descriptorType = descriptorType;
                (resultBuffer.end()-1)->selectorType = selectorType;

                // store number of descriptor matches in result buffer (for all frames except for the first frame)
                (resultBuffer.end()-1)->numDescMatches = matches.size();

                // clear temporary variables that are no longer needed (avoid memory leaks)
                matches.clear();

                // visualize matches between the current and the previous image
                if (bVisDebug)
                {
                    string windowName = "Matching keypoints between two camera images";
                    showKptMatches(
                        (dataBuffer.end()-2)->keypoints, (dataBuffer.end()-1)->keypoints,
                        (dataBuffer.end()-2)->cameraImgGray, (dataBuffer.end()-1)->cameraImgGray,
                        (dataBuffer.end()-1)->kptMatches, bWait, windowName);
                }
            }
            else
            {
                // inform user about incompatibility of "KAZE" descriptor extractor with keypoint detectors other than "KAZE" or "AKAZE"
                cout << "Descriptor extractor type "
                    << descExtractorType
                    << " is not compatible with keypoint detector type "
                    << detectorType
                    << endl;
                cout << "... so this combination will be skipped" << endl;

                // store configuration for keypoint matching in result buffer (for all frames except for the first frame)
                (resultBuffer.end()-1)->matcherType = matcherType;  // store selected matcher type in result buffer
                (resultBuffer.end()-1)->descriptorType = descriptorType;  // store selected descriptor type in result buffer
                (resultBuffer.end()-1)->selectorType = selectorType;  // store selected selector type in result buffer
                (resultBuffer.end()-1)->numDescMatches = 0;  // store number of descriptor matches in result buffer
            }
            
            // store processing time for keypoint descriptor matching in result buffer (for all frames except for the first frame)
            (resultBuffer.end()-1)->t_matchDescriptors = t_matchDescriptors; // is zero for skipped combinations
            
            // store the cumulated processing time for keypoint detection, descriptor extraction and descriptor matching 
            // in result buffer (for all frames except for the first frame)
            (resultBuffer.end()-1)->t_sum_det_desc_match = t_detKeypoints + t_descKeypoints + t_matchDescriptors;

            // EOF #7: MATCH KEYPOINT DESCRIPTORS
            cout << "#7: MATCH KEYPOINT DESCRIPTORS done" << endl;

            
            /* #8: TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            map<int, int> bbBestMatches;
            if (bSkipCombination == false)
            {
                // associate bounding boxes between current and previous frame using keypoint matches
                // NOTE: Some false positive keypoint matches that are not associated with the right target bounding boxes may slip through!
                matchBoundingBoxes((dataBuffer.end()-1)->kptMatches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1));
            }
            else
            {
                // skip bounding box matching for incompatible keypoint detector - descriptor extractor combinationss()
                cout << "Skip bounding box matching due to incompatible keypoint detector - descriptor extractor combination" << endl;
                cout << "bbBestMatches are empty" << endl;
            }
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;

            // EOF #8: TRACK 3D OBJECT BOUNDING BOXES
            cout << "#8: TRACK 3D OBJECT BOUNDING BOXES done" << endl;


            /* #9: COMPUTE TTC ON OBJECT IN FRONT */

            // initialize number of Lidar points on target
            int numLidarPointsOnTarget = 0;

            // initialize number of keypoint matches on target
            int numKptMatchesOnTarget = 0;

            // initialize time-to-collision for Lidar and Camera
            double ttcLidar = NEG_INF;  // set to negative infinity => no collision
            double ttcCamera = NEG_INF; // set to negative infinity => no collision

            if (bSkipCombination == false)
            {
                // print previous and current file name
                if (bPrintDebugInfo)
                {
                    cout << "Measure time to collision from 3D Lidar measurements and 2D keypoint matches:" << endl;
                    cout << "Previous image frame = " << (dataBuffer.end()-2)->imgFilename << endl;
                    cout << "Current image frame  = " << (dataBuffer.end()-1)->imgFilename  << endl;
                }

                // loop over all BB match pairs
                for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
                {
                    // find bounding boxes in the current frame associated with the current BB match pair
                    BoundingBox *prevBB, *currBB;
                    for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                    {
                        if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB in the current frame
                        {
                            currBB = &(*it2);
                        }
                    }

                    // find bounding boxes in the previous frame associated with the current BB match pair
                    for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                    {
                        if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB in the previous frame
                        {
                            prevBB = &(*it2);
                        }
                    }

                    // print previous and current target bounding box ID
                    if (bPrintDebugInfo)
                    {
                        cout << "Matched target bounding box IDs from previous and current frame:" << endl;
                        cout << "Target BB ID from previous frame = " << prevBB->boxID << endl;
                        cout << "Target BB ID from current frame  = " << currBB->boxID  << endl;
                    }

                    // compute TTC for current match
                    /*
                    * Please note:
                    *   The 3D Lidar points have been cropped from the BB ROI of the next target object on ego lane! All other Lidar points
                    *   have been dropped. Therefore, we do not get any time to collision measurements to the other object detections.
                    */
                    if( currBB->lidarPoints.size() > 0 && prevBB->lidarPoints.size() > 0 ) // only compute TTC if we have Lidar points
                    {
                        //// STUDENT ASSIGNMENT
                        /* options how to estimate a (robust) distance estimate from the Lidar points in the ROI
                        *  TTCLidarEstimationOption = 1: closest distance
                        *  TTCLidarEstimationOption = 2: mean distance
                        *  TTCLidarEstimationOption = 3: closest distance larger than mean distance minus x-times standard deviation
                        *  TTCLidarEstimationOption = 4: median distance
                        *  TTCLidarEstimationOption = 5: threshold on sorted distances
                        */

                        //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                        computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar, TTCLidarEstimationOption,
                            *currBB, P_rect_00, R_rect_00, RT, & (dataBuffer.end() - 1)->cameraImg, bVisDebug, bWait);
                        
                        // get number of Lidar points associated with the target bounding box
                        numLidarPointsOnTarget = currBB->lidarPoints.size();
                        //// EOF STUDENT ASSIGNMENT

                        //// STUDENT ASSIGNMENT
                        //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                        clusterKptMatchesWithROI(*prevBB, *currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, 
                            (dataBuffer.end() - 1)->kptMatches);

                        //// TASK FP.4 -> compute time-to-collision based on camera data (implement -> computeTTCCamera)
                        computeTTCCamera(*prevBB, *currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, 
                            currBB->kptMatches, sensorFrameRate, ttcCamera, & (dataBuffer.end() - 1)->cameraImg, bVisDebug, bWait,
                            bPrintDebugInfo);

                        // get number of keypoint matches associated with the target bounding box
                        numKptMatchesOnTarget = currBB->kptMatches.size();
                        //// EOF STUDENT ASSIGNMENT

                        if (bVisDebug)
                        {
                            // show 3D Lidar objects from previous frame (top view projection)
                            string windowName_1 = "3D Lidar objects (previous frame)";
                            std::vector<BoundingBox> prevBBList = {*prevBB};
                            show3DObjects(prevBBList, cv::Size(4.0, 20.0), cv::Size(1000, 1000), bWait, windowName_1);
                        
                            // show 3D Lidar objects from current frame (top view projection)
                            string windowName_2 = "Lidar points in ROI (current frame)";
                            std::vector<BoundingBox> currBBList = {*currBB};
                            show3DObjects(currBBList, cv::Size(4.0, 20.0), cv::Size(1000, 1000), bWait, windowName_2);
                        
                            // show all 2D keypoints and 2D keypoint matches associated with the target bounding box in previous and curent image frame
                            string windowName_3 = "All keypoints and matching keypoints in ROI between previous frame (left) and current frame (right)";
                            showKptMatchesWithROI(
                                (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                (dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 1)->cameraImg, currBB->kptMatches,
                                prevBB->roi, currBB->roi, bWait, windowName_3);
                        }

                        if (bVis)
                        {
                            // show 3D Lidar points and 2D keypoints associated with the target bounding box overlayed on 2D image plane in the current frame
                            string windowName = "Final Results : TTC (current frame: " + (dataBuffer.end()-1)->imgFilename + ")";
                            cv::Mat visImgCurr = (dataBuffer.end() - 1)->cameraImg.clone();
                            cv::drawKeypoints((dataBuffer.end() - 1)->cameraImg, currBB->keypoints, visImgCurr,
                                cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                            showLidarImgOverlay(visImgCurr, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImgCurr, false, windowName);
                            // add target object bounding box (= ROI)
                            cv::rectangle(visImgCurr, cv::Point(currBB->roi.x, currBB->roi.y), 
                                cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                            // add time-to-collision calculation results to the image
                            char str[200];
                            sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                            putText(visImgCurr, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));
                            cv::namedWindow(windowName, cv::WINDOW_NORMAL); // cv::WINDOW_NORMAL = 1 the user can resize the window (no constraint) 
                            cv::imshow(windowName, visImgCurr);
                            cout << "Press key to continue to next frame" << endl;
                            cv::waitKey(0); // wait for key to be pressed
                            if (bSaveImageToFile)
                            {
                                // try to save the result image file
                                try
                                {                                    
                                    // save current image to result directory
                                    string final_result_image_name = "final_results_ttc_" + (dataBuffer.end()-1)->imgFilename;
                                    cv::imwrite(resultBasePath + final_result_image_name, visImgCurr);
                                    cout << "Final results TTC image saved as " << resultBasePath + final_result_image_name << endl;
                                }
                                catch(const exception& e)
                                {
                                    // show exeption and return 1
                                    cerr << e.what() << endl;
                                    return 1;
                                }
                            }
                            //cv::destroyWindow(windowName); // close window
                        }
                    }
                } // eof TTC computation
            } // eof loop over all BB matches
            else
            {
                // skip time-to-collision estimation for incompatible keypoint detector - descriptor extractor combinations
                cout << "Skip time-to-collision estimation due to incompatible keypoint detector - descriptor extractor combination" << endl;
                cout << "The number of Lidar points and the number of keypoint matches on target are all zero" << endl;
                cout << "Time-to-collision values are all zero" << endl;            
            }
            
            // store time-to-collision estimation results based on Lidar data in result buffer (for all frames except for the first frame)
            (resultBuffer.end()-1)->numLidarPointsOnTarget = numLidarPointsOnTarget;
            (resultBuffer.end()-1)->ttcLidar = ttcLidar;
             
            // store time-to-collision estimation results based on camera data in result buffer (for all frames except for the first frame)
            (resultBuffer.end()-1)->numKptMatchesOnTarget = numKptMatchesOnTarget;
            (resultBuffer.end()-1)->ttcCamera = ttcCamera;
            
            // EOF #9: COMPUTE TTC ON OBJECT IN FRONT
            cout << "#9: COMPUTE TTC ON OBJECT IN FRONT done" << endl;

        }
        else
        {
            // store results for the first frame in result buffer (exception)
            (resultBuffer.end()-1)->matcherType = matcherType;
            (resultBuffer.end()-1)->descriptorType = descriptorType;
            (resultBuffer.end()-1)->selectorType = selectorType;
            (resultBuffer.end()-1)->numDescMatches = 0;
            (resultBuffer.end()-1)->t_matchDescriptors = 0.0;
            (resultBuffer.end()-1)->t_sum_det_desc_match = t_detKeypoints + t_descKeypoints;
            (resultBuffer.end()-1)->numLidarPointsOnTarget = 0;
            (resultBuffer.end()-1)->ttcLidar = NEG_INF;
            (resultBuffer.end()-1)->numKptMatchesOnTarget = 0;
            (resultBuffer.end()-1)->ttcCamera = NEG_INF;
        }

    } // eof loop over all images

    // push current result buffer into the vector of result buffers
    evalResultBuffers.push_back(resultBuffer);

    // #10: Export results to csv file
    if (bExportResultsToCSV)
    {
        // define filepath and filename for result file
        string resultFilepath = resultBasePath;
        string resultFilename = "3D_object_tracking_using_";
        string resultFiletype = ".csv";
        string resultFullFilename = resultFilepath
                                + resultFilename
                                + resultBuffer.begin()->detectorType
                                + "_and_"
                                + resultBuffer.begin()->descExtractorType
                                + resultFiletype;
        cout << "Export evaluation results to " << resultFullFilename << endl;

        // try to export the results to csv file
        try
        {
            // export results to csv file
            exportResultsToCSV(resultFullFilename, resultBuffer, bPrintDebugInfo);
        }
        catch(const exception& e)
        {
            // show exeption and return 1
            cerr << e.what() << endl;
            return 1;
        }

        // EOF #10: EXPORT RESULTS TO CSV FILE
        cout << "#10: EXPORT RESULTS TO CSV FILE done" << endl;
    }

    // return 0 if program terminates without errors
    return 0;
}


/* MAIN FUNCTION */
int main(int argc, const char *argv[])
{
    /* SWITCH BETWEEN SINGLE RUN (USING MANUAL CONFIGURATION) AND BATCH RUN (EVALUATING DIFFERENT DETECTOR EXTRACTOR COMBINATIONS) */

    // Select batch run or single run mode
    bool bBatchMode = false; // options: true => batch run; false => single run

    if (!bBatchMode)
    {
        /* --- SINGLE RUN MODE --- */

        // print selected run mode
        cout << "Evaluation of 3D object tracking in single run mode:" << endl;
                
        // ringbuffer implementation based on boost circular buffer => check installed boost version
        cout << "Using Boost version "
            << BOOST_VERSION / 100000     << "."  // major version
            << BOOST_VERSION / 100 % 1000 << "."  // minor version
            << BOOST_VERSION % 100                // patch level
            << endl;
        
        /* MANUAL CONFIGURATION FOR 3D OBJECT TRACKING STUDENT ASSIGNMENT */

        // Select keypoint detector type
        string detectorType = "FAST";  // options: "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "KAZE", "AKAZE", "SIFT", "SURF"

        // optional : limit number of keypoints (helpful only for debugging and learning => Do not use in real application!)
        bool bLimitKpts = false;
        int maxKpts = 200;  // only for testing => Do not limit keypoints in real application!
        
        // select keypoint descriptor extractor type ("KAZE" and "AKAZE" only work with "KAZE" or "AKAZE" keypoints)
        string descExtractorType = "BRIEF";  // options: "BRISK", "BRIEF", "ORB", "FREAK", "KAZE", "AKAZE", "SIFT", "SURF"

        //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
        //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t = 0.8 in file matching2D.cpp

        // select descriptor matcher tpye
        string matcherType = "MAT_FLANN";  // options: MAT_BF, MAT_FLANN

        // select descriptor type (use "DES_HOG" for "KAZE", "SIFT" and "SURF", otherwise use "DES_BINARY")
        string descriptorType = "DES_BINARY";  // options: DES_BINARY, DES_HOG

        // select selector type
        string selectorType = "SEL_KNN";  // SEL_NN, SEL_KNN

        /* options how to estimate a (robust) distance estimate from the Lidar points in the ROI
        *  TTCLidarEstimationOption = 1: closest distance
        *  TTCLidarEstimationOption = 2: mean distance
        *  TTCLidarEstimationOption = 3: closest distance larger than mean distance minus x-times standard deviation
        *  TTCLidarEstimationOption = 4: median distance
        *  TTCLidarEstimationOption = 5: threshold on sorted distances
        */
        int TTCLidarEstimationOption = 4;

        // visualization and printout (debugging)
        bool bVis = true;             // visualize results
        bool bVisDebug = true;        // visualize intermediate results (for debugging or to look into details)
        bool bPrintDebugInfo = false; // print additional information for debugging
        bool bWait = true;            // wait for keypress to continue after each plot => close the plot on keypress
        
        // save images to file
        bool bSaveImageToFile = true;
        
        // export evaluation results to csv file
        bool bExportResultsToCSV = true;
        
        // initialize (empty) vector of evaluation result buffers
        vector<boost::circular_buffer<EvalResults>> evalResultBuffers;

        // try to run track3DObjects in single run mode
        try
        {
            // evaluate 3D object tracking performance
            if (track3DObjects(
                evalResultBuffers,
                detectorType,
                bLimitKpts,
                maxKpts,
                descExtractorType,
                matcherType,
                descriptorType,
                selectorType,
                TTCLidarEstimationOption,
                bVis,
                bVisDebug,
                bPrintDebugInfo,
                bWait,
                bSaveImageToFile,
                bExportResultsToCSV) == 0)
            {
                // return 0 if program terminates without errors
                return 0;
            }
            else
            {
                // return 1 if program terminates with errors
                return 1;
            }
        }
        catch(const exception& e)
        {
            // show exeption and return 1
            cerr << e.what() << endl;
            return 1;
        }

    }
    else
    {
        /* --- BATCH RUN MODE --- */

        // print selected run mode
        cout << "Evaluation of 3D object tracking in batch run mode:" << endl;
        
        // vector of keypoint detector types to evaluate
        vector<string> vec_detectorTypes = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "KAZE", "AKAZE", "SIFT", "SURF"};
        
        // optional : limit number of keypoints (helpful only for debugging and learning => Do not use in real application!)
        bool bLimitKpts = false;
        int maxKpts = 200;
        
        // vector of keypoint descriptor extractor types to evaluate ("KAZE" and "AKAZE" only work with "KAZE" or "AKAZE" keypoints)
        vector<string> vec_descExtractorTypes = {"BRISK", "BRIEF", "ORB", "FREAK", "KAZE", "AKAZE", "SIFT", "SURF"};
        
        // set descriptor matcher tpye
        //string matcherType = "MAT_BF";  // options: MAT_BF, MAT_FLANN
        string matcherType = "MAT_FLANN";

        // set binary descriptor type => will be automatically adapted to the descriptor extractor type in the loop over all combinations
        // select descriptor type (use "DES_HOG" for "KAZE", "SIFT" and "SURF", otherwise use "DES_BINARY")
        string descriptorType = "DES_BINARY";  // options: DES_BINARY, DES_HOG    
        
        // set selector type
        string selectorType = "SEL_KNN";  // SEL_NN, SEL_KNN

        /* options how to estimate a (robust) distance estimate from the Lidar points in the ROI
        *  TTCLidarEstimationOption = 1: closest distance
        *  TTCLidarEstimationOption = 2: mean distance
        *  TTCLidarEstimationOption = 3: closest distance larger than mean distance minus x-times (x = 1.0) standard deviation {x = 0.0 ... 3.0 or 4.0}
        *  TTCLidarEstimationOption = 4: median distance
        *  TTCLidarEstimationOption = 5: threshold (= 0.33) on sorted distances {threshold = 0: closest point | 0.5: median | 1.0: farthest point} 
        */
        int TTCLidarEstimationOption = 4;
        
        // visualization and printout are switched off in batch mode
        bool bVis = false;            // visualization of keypoint matching results
        bool bVisDebug = false;       // visualize intermediate results (for debugging or to look into details)
        bool bPrintDebugInfo = false; // print additional information for debugging
        bool bWait = false;           // wait for keypress to continue after each plot => close the plot on keypress

        // save images to file is switched off in batch mode
        bool bSaveImageToFile = false; // save image to file off
        
        // export evaluation results of each keypoint detector - descriptor extractor combination to csv file
        bool bExportResultsToCSV = true;
        
        // initialize vector of evaluation result buffers
        vector<boost::circular_buffer<EvalResults>> evalResultBuffers;

        // iterator
        int itr = 0;

        for (vector<string>::const_iterator ptrDetType = vec_detectorTypes.begin(); ptrDetType != vec_detectorTypes.end(); ptrDetType++)
        {
            for (vector<string>::const_iterator ptrDescExtType = vec_descExtractorTypes.begin(); ptrDescExtType != vec_descExtractorTypes.end(); ptrDescExtType++)
            {
                // print current iterator
                cout << "\nIteration no. " << itr++ << endl;

                if ((*ptrDescExtType) == "KAZE" || (*ptrDescExtType) == "SIFT" || (*ptrDescExtType) == "SURF")
                {
                    // use gradient based descriptor type for SIFT and SURF
                    descriptorType = "DES_HOG";
                }
                else
                {
                    // use binary descriptor type for all other descriptor extractor types
                    descriptorType = "DES_BINARY";
                }
                
                // print current configuration
                cout << "\n" << "Next configuration for 2D feature tracking:" << endl;
                cout << "Feature detector type     = " << (*ptrDetType) << endl;
                cout << "Descriptor extractor type = " << (*ptrDescExtType) << endl;
                cout << "Matcher type              = " << matcherType << endl;
                cout << "Descriptor type           = " << descriptorType << endl;
                cout << "Selector type             = " << selectorType << "\n" << endl;

                // KADZE and AKAZE feature extractors only work with KAZE or AKAZE keypoints
                // => skip other configurations

                // try to run 3D object tracking in batch run mode
                try
                {
                    // evaluate 3D object tracking performance in batch mode
                    if (track3DObjects(
                        evalResultBuffers,
                        (*ptrDetType),
                        bLimitKpts,
                        maxKpts,
                        (*ptrDescExtType),
                        matcherType,
                        descriptorType,
                        selectorType,
                        TTCLidarEstimationOption,
                        bVis,
                        bVisDebug,
                        bPrintDebugInfo,
                        bWait,
                        bSaveImageToFile,
                        bExportResultsToCSV) == 0)
                    {
                        continue;
                    }
                    else
                    {
                        // return 1 if program terminates with errors
                        return 1;
                    }
                }
                catch(const exception& e)
                {
                    // show exeption and return 1
                    cerr << e.what() << endl;
                    return 1;
                }

                // wait for user key press
                string tmp;
                cout << "Press any key to continue: ";
                cin >> tmp;
                cout << "endl";
            }
        }

        // export overall results in an overview on all keypoint detector - descriptor extractor combinations to a csv file
        if (bExportResultsToCSV)
        {
            // define filepath and filename for result file
            string resultFilepath = resultBasePath;
            string resultFilename = "3D_object_tracking_overall_results";
            string resultFiletype = ".csv";
            string resultFullFilename = resultFilepath
                                    + resultFilename
                                    + resultFiletype;
            cout << "Export overall evaluation results to " << resultFullFilename << endl;

            // try to export the results to csv file
            try
            {
                // export overall results to csv file
                exportOverallResultsToCSV(resultFullFilename, evalResultBuffers, bPrintDebugInfo);
            }
            catch(const exception& e)
                {
                // show exeption and return 1
                cerr << e.what() << endl;
                return 1;
            }

            cout << "#6 : EXPORT OVERALL RESULTS TO CSV FILE done" << endl;
        }

        // return 0 if program terminates without errors
        return 0;   
    }
}