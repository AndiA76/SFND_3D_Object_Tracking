// ============================================================================
//  
//  Project 2.2: 3D Object Tracking (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
//  
//  Source:      https://github.com/udacity/SFND_3D_Object_Tracking
//
// ============================================================================

// function definitions for 2D keypoint detection and 2D feature matching

#include <numeric>
#include <opencv2/highgui.hpp>
#include "matching2D.hpp"

using namespace std;


// show keypoint matches between two image frames
void showKptMatches(
    std::vector<cv::KeyPoint> & kPtsSource, std::vector<cv::KeyPoint> & kPtsRef, 
    cv::Mat & kPtsSourceImage, cv::Mat & kPtsRefImage, std::vector<cv::DMatch> & kPtMatches,
    bool bWait, std::string windowName)
{
    // plot keypoint matches between image 1 and image 2
    cv::Mat kPtMatchImg = kPtsRefImage.clone();
    cv::drawMatches(
        kPtsSourceImage, kPtsSource, kPtsRefImage, kPtsRef, kPtMatches, kPtMatchImg,
        cv::Scalar::all(-1), cv::Scalar::all(-1),
        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //string windowName = "Matching keypoints between two camera images";
    //cv::namedWindow(windowName, cv::WINDOW_GUI_EXPANDED); // cv::WINDOW_GUI_EXPANDED = 7 status bar and tool bar
    cv::namedWindow(windowName, cv::WINDOW_NORMAL); // cv::WINDOW_NORMAL = 1 the user can resize the window (no constraint) 
    cv::imshow(windowName, kPtMatchImg);
    if (bWait)
    {
        cout << "Press key to continue" << endl;
        cv::waitKey(0); // wait for key to be pressed
        cv::destroyWindow(windowName); // close window
    }
}


// show keypoint matches between two image frames adding a rectangular bouding box as region of interest (ROI)
void showKptMatchesWithROI(
    std::vector<cv::KeyPoint> & kptsSource, std::vector<cv::KeyPoint> & kptsRef, 
    cv::Mat & kptsSourceImage, cv::Mat & kptsRefImage, std::vector<cv::DMatch> & kPtMatches,
    cv::Rect & kptsSourceROI, cv::Rect & kptsRefROI, bool bWait, std::string windowName)
{
    // plot keypoint matches between image 1 and image 2
    cv::Mat kPtMatchImg = kptsRefImage.clone();
    // get width of the reference image (which must be the same size as the source image)
    cv::Size kPtMatchImg_size = kPtMatchImg.size();
    cv::drawMatches(
        kptsSourceImage, kptsSource, kptsRefImage, kptsRef, kPtMatches, kPtMatchImg,
        cv::Scalar::all(-1), cv::Scalar::all(-1),
        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // add an ROI bounding box to the source frame (left side) in red and to the reference frame (right side) in green
    cv::rectangle(kPtMatchImg, cv::Point(kptsSourceROI.x, kptsSourceROI.y),
        cv::Point(kptsSourceROI.x + kptsSourceROI.width, kptsSourceROI.y + kptsSourceROI.height),
        cv::Scalar(0, 0, 255), 2);
    cv::rectangle(kPtMatchImg, cv::Point(kPtMatchImg_size.width + kptsRefROI.x, kptsRefROI.y), 
        cv::Point(kPtMatchImg_size.width + kptsRefROI.x + kptsRefROI.width, kptsRefROI.y + kptsRefROI.height),
        cv::Scalar(0, 255, 0), 2);
    //string windowName = "Matching keypoints between two camera images";
    //cv::namedWindow(windowName, cv::WINDOW_GUI_EXPANDED); // cv::WINDOW_GUI_EXPANDED = 7 status bar and tool bar
    cv::namedWindow(windowName, cv::WINDOW_NORMAL); // cv::WINDOW_NORMAL = 1 the user can resize the window (no constraint) 
    cv::imshow(windowName, kPtMatchImg);
    if (bWait)
    {
        cout << "Press key to continue" << endl;
        cv::waitKey(0); // wait for key to be pressed
        cv::destroyWindow(windowName); // close window
    }
}


// find best matches for keypoints in two camera images based on several matching methods
double matchDescriptors(
    std::vector<cv::KeyPoint> & kptsSource, std::vector<cv::KeyPoint> & kptsRef,
    cv::Mat & descSource, cv::Mat & descRef, std::vector<cv::DMatch> & matches,
    std::string descriptorType, std::string matcherType, std::string selectorType)
{
    /* matchDescriptors() computes the best keypoint matchtes between a source image (image 1) and a reference
    *  image (image 2) using either brute force matching, nearest neighbor matching or k-nearest neighbor 
    *  matching (k = 2), where the latter is combined with a distance ratio test. The function returns a vector
    *  of the best matches including the following items of type DMatch:
    *  
    *  - DMatch.distance - Distance between descriptors in the two images. The lower, the better it is.
    *  - DMatch.trainIdx - Index of the descriptor in train descriptors (here, it’s the list of descriptors in
    *    the reference image, or image 2, respectively).
    *  - DMatch.queryIdx - Index of the descriptor in query descriptors (here, it's the list of descriptors in
    *    the source image, or image 1, respectively).
    *  - DMatch.imgIdx - Index of the train image, or reference image (image 2), respectively.
    */

    // init and configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    { // use brute-force matching

        // select ditance norm type to compare the descriptors
        //int normType = cv::NORM_HAMMING;
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;

        // brute force matching approach searching through all avaialable keypoints and keypoint descriptors 
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "Use BF matching: BF cross-check = " << crossCheck << endl;

    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    { // use FLANN-based matching

        // workaround for BUG in OpenCV
        if (descSource.type() != CV_32F || descRef.type()!=CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        // efficient FLANN-based matching using a KD-tree to quickly search through available keypoints and descriptors
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cout << "Use FLANN matching" << endl;
    }

    // create variable to hold the processing time for keypoint descriptor matching
    double t = 0.0;

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    {
        // perform nearest neighbor matching (best match): yields only one best match (timed process)
        t = (double)cv::getTickCount(); // trigger timer
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); // stop timer
        cout << " NN matching with n = "
            << matches.size()
            << " matches in "
            << 1000 * t / 1.0
            << " ms"
            << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {
        // perform k nearest neighbor matching (k=2): yields k best matches, here: k = 2 (timed process)
        vector<vector<cv::DMatch>> knn_matches;
        t = (double)cv::getTickCount(); // trigger timer
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the two (k = 2) best matches
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); // stop timer
        cout << "KNN matching with n = " << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        /* NOTE: 
        *  Descriotor distance ratio test versus cross-check matching:
        *  In general, the descriotor distance ratio test is less precise, but more efficient than cross-check
        *  matching. 
        *  In cross-check matching, the keypoint matching is done twice (image1 -> image 2 and vice versa), and
        *  keypoints are only accepted when keypoint matches found in image 2 for keypoints from image 1 match
        *  with keypoint matches found in image 1 for keypoints from image 2. As this needs more processing time
        *  the distance ratio test is preferred in this task.
        */

        // filter out ambiguous matches using descriptor distance ratio test and reduce the number of false positives
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        cout << "Total number of keypoints matches = " << matches.size() << endl;
        cout << "Number of keypoints removed = " << knn_matches.size() - matches.size() << endl;
    }

    // return processing time for keypoint descriptor matching
    return t;
}


// use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(
    vector<cv::KeyPoint> & keypoints, cv::Mat & img, cv::Mat & descriptors, string descExtractorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descExtractorType.compare("BRISK") == 0)
    {
        /* BRISK (Binary robust invariant scalable keypoints) feature detector and descriptor extractor
        *  OpenCV: https://docs.opencv.org/4.1.2/de/dbf/classcv_1_1BRISK.html
        */

        // set BRISK descriptor extractor parameters
        int threshold = 30; // FAST/AGAST detection threshold score
        int octaves = 3; // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint

        // create BRISK descriptor extractor
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descExtractorType.compare("BRIEF") == 0)
    {
        /* BRIEF (Binary robust independent elementary features)
        *  OpenCV: https://docs.opencv.org/4.1.2/d1/d93/classcv_1_1xfeatures2d_1_1BriefDescriptorExtractor.html
        */

        // set BRIEF descriptor extractor parameters
        int bytes = 32; // length of the descriptor in bytes, valid values are: 16, 32 (default) or 64
		bool use_orientation = false; // sample patterns using keypoints orientation, disabled by default

        // create BRIEF descriptor extractor
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
    }
    else if (descExtractorType.compare("ORB") == 0)
    {
        /* ORB (Oriented FAST and Rotated BRIEF)
        *  OpenCV: https://docs.opencv.org/4.1.2/db/d95/classcv_1_1ORB.html
        */

        // set ORB descriptor extractor parameters
        int nfeatures = 500; // maximum number of features to retain
		float scaleFactor = 1.2f; // pyramid decimation ratio, greater than 1 (scaleFactor==2 => classical pyramid)
        int nlevels = 8; // number of pyramid levels (smallest level has linear size equal to input_image_linear_size/pow(scaleFactor, nlevels - firstLevel))
		int edgeThreshold = 31; // size of the border where the features are not detected (should roughly match the patchSize parameter)
		int firstLevel = 0; // level of pyramid to put source image to (Previous layers are filled with upscaled source image)
		int WTA_K = 2; // number of points that produce each element of the oriented BRIEF descriptor (default value: 2, other possible values: 3, 4)
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;  // default: use HARRIS_SCORE to rank features (or: FAST_SCORE => slightly less stable keypoints, but faster)
		int patchSize = 31; // size of the patch used by the oriented BRIEF descriptor
		int fastThreshold = 20; // the fast threshold

        // create ORB descriptor extractor
		extractor = cv::ORB::create(
            nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel,
            WTA_K, scoreType, patchSize, fastThreshold);
	}
    else if (descExtractorType.compare("FREAK") == 0)
    {
        /* FREAK (Fast Retina Keypoint) descriptor extractor
        *  https://docs.opencv.org/4.1.2/df/db4/classcv_1_1xfeatures2d_1_1FREAK.html
        *  Remark: FREAK is only a keypoint descriptor! => Another feature detector is needed to find the keypoints.
        */

        // set FREAK descriptor extractor parameters
        bool orientationNormalized = true; // enable orientation normalization
		bool scaleNormalized = true; // enable scale normalization
		float patternScale = 22.0f; // scaling of the description pattern
		int nOctaves = 4; // number of octaves covered by the detected keypoints
		// const std::vector< int > & selectedPairs = std::vector< int >();  // (optional) user defined selected pairs indexes

        // create FREAK descriptor extractor
        extractor = cv::xfeatures2d::FREAK::create(
            orientationNormalized, scaleNormalized, patternScale, nOctaves);
    }
    else if (descExtractorType.compare("KAZE") == 0)
	{
        /* KAZE feature detector and descriptor extractor
        *  OpenCV: https://docs.opencv.org/4.1.2/d3/d61/classcv_1_1KAZE.html
        */

        // set KAZE descriptor extractor parameters
        bool extended = false; // set to enable extraction of extended (128-byte) descriptor.
		bool upright = false; // set to enable use of upright descriptors (non rotation-invariant).
		float threshold = 0.001f; // detector response threshold to accept point
		int nOctaves = 4; // maximum octave evolution of the image
		int nOctaveLayers = 4; // Default number of sublevels per scale level
		cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2; // options: DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER

        // create KAZE descriptor extractor
        extractor = cv::KAZE::create(
            extended, upright, threshold, nOctaves, nOctaveLayers, diffusivity); 	
	}
	else if (descExtractorType.compare("AKAZE") == 0)
	{
        /* AKAZE (Accelerated-KAZE) feature detector and descriptor extractor
        *  OpenCV: https://docs.opencv.org/4.1.2/d8/d30/classcv_1_1AKAZE.html
        */

        // set AKAZE descriptor extractor parameters
        cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB; // options: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
		int descriptor_size = 0; // size of the descriptor in bits. 0 -> full size
		int descriptor_channels = 3; // number of channels in the descriptor (1, 2, 3)
		float threshold = 0.001f; // detector response threshold to accept point
		int nOctaves = 4; // maximum octave evolution of the image
		int nOctaveLayers = 4; // Default number of sublevels per scale level
		cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2; // options: DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER
        
        // create AKAZE descriptor extractor
        extractor = cv::AKAZE::create(
            descriptor_type, descriptor_size, descriptor_channels, threshold,
            nOctaves, nOctaveLayers, diffusivity);
	}
    else if (descExtractorType.compare("SIFT") == 0)
    {
        /* SIFT (Scale Invariant Feature Transform) feature detector and descriptor extractor
        *  https://docs.opencv.org/4.1.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
        */
        
        // set SIFT descriptor extractor parameters
        int nfeatures = 0; // number of best features to retain (features are ranked by their scores measured as the local contrast)
        int nOctaveLayers = 3; // number of layers in each octave (3 is the value used in D. Lowe paper)
        double contrastThreshold = 0.04; // contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions
        double edgeThreshold = 10; // threshold used to filter out edge-like features
        double sigma = 1.6; // sigma of the Gaussian applied to the input image at the octave #0

        // create SIFT descriptor extractor
    	extractor = cv::xfeatures2d::SIFT::create(
            nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }
    else if (descExtractorType.compare("SURF") == 0)
    {
        /* SURF (Speeded-up robust features) feature detector and descriptor extractor
        *  OpenCV: https://docs.opencv.org/4.1.2/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html#details
        */

        // set SURF descriptor extractor parameters
        double hessianThreshold = 100; // threshold for hessian keypoint detector used in SURF
		int nOctaves = 4; // number of pyramid octaves the keypoint detector will use
		int nOctaveLayers = 3; // number of octave layers within each octave
		bool extended = false; // extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors)
		bool upright = false; // up-right or rotated features flag (true - do not compute orientation of features; false - compute orientation)

        // create SURF descriptor extractor
        extractor = cv::xfeatures2d::SURF::create(
            hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
    }
    else {
        // throw error message
        throw "Error: Wrong input argument to descKeypoints(): Feature descriptor (extractor) type not defined!";
	}

    // perform feature description (timed process)
    double t = (double)cv::getTickCount();  // trigger timer
    extractor->compute(img, keypoints, descriptors);  // extract feature descriptors
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();  // stop timer
    cout << descExtractorType
        << " descriptor extraction for n = "
        << keypoints.size()
        << " keypoints in "
        << 1000 * t / 1.0
        << " ms"
        << endl;

    // return processing time for keypoint descriptor extraction
    return t;
}


// detect keypoints in image using the traditional Shi-Tomasi corner detector (based in image gradients, slow)
double detKeypointsShiTomasi(vector<cv::KeyPoint> & keypoints, cv::Mat & img, bool bVis, bool bWait)
{
    /* Shi-Tomasi corner detector
    *  OpenCV: https://docs.opencv.org/4.1.2/d8/dd8/tutorial_good_features_to_track.html
    */

    // compute Shi-Tomasi detector parameters based on image size
    int blockSize = 4; //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize; // minimum possible Euclidean distance between the returned corners
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints
    bool useHarrisDetector = false; // parameter indicating whether to use a Harris detector or cornerMinEigenVal
    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04; // free parameter of the Harris detector

    // apply corner detection
    double t = (double)cv::getTickCount(); // trigger timer
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(
        img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); // stop timer
    cout << "Shi-Tomasi feature detection with n="
        << keypoints.size()
        << " keypoints in "
        << 1000 * t / 1.0
        << " ms"
        << endl;

    // visualize results
    if (bVis)
    {
        // plot image with keypoints
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(
            img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, cv::WINDOW_NORMAL); // cv::WINDOW_NORMAL = 1 enables you to resize the window
        imshow(windowName, visImage);
        if (bWait)
        {
            cout << "Press key to continue" << endl;
            cv::waitKey(0); // wait for key to be pressed
            cv::destroyWindow(windowName); // close window
        }
    }

    // return processing time for keypoint detection
    return t;
}


// detect keypoints in image using the traditional Harris corner detector
double detKeypointsHarris(vector<cv::KeyPoint> & keypoints, cv::Mat & img, bool bVis, bool bWait)
{
    /* Harris corner detector
    *  OpenCV: https://docs.opencv.org/4.1.2/d4/d7d/tutorial_harris_detector.html
    *          https://docs.opencv.org/master/dc/d0d/tutorial_py_features_harris.html
    */

    // set Harris corner detector parameters
    int blockSize = 2; // for every pixel, a blockSize × blockSize neighborhood is considered for corner detection
    int apertureSize = 3; // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04; // Harris parameter (see equation for details)

    // apply Harris corner detector and normalize output
    double t = (double)cv::getTickCount(); // trigger timer
    cv::Mat dst, dst_norm, dst_norm_scaled; // define destination matrices for intermediate and final results
    dst = cv::Mat::zeros(img.size(), CV_32FC1); // initialize final result matrix
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT); // detect Harris corners
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat()); // normalize input array
    cv::convertScaleAbs(dst_norm, dst_norm_scaled); // scales, calculates absolute values, and converts the result to 8-bit

    /* Locate local maxima in Harris corner detector response map and perform non-maximum suppression 
    *  in the local neighborhood around each maximum and store the resulting keypoint coordinates in a
    *  list of keypoints of type vector<cv::KeyPoint> (s. input argument)
    */

    // set maximum permissible overlap between two features in %, used during non-maxima suppression
    double maxOverlap = 0.0;

    // loop over all rows and colums of Harris corner detector response map
    for (size_t j = 0; j < dst_norm.rows; j++)
    { // loop over all rows
    
        for (size_t i = 0; i < dst_norm.cols; i++)
        { // loop over all cols

            // get response from normalized Harris corner detection response matrix scaled to positive 8 bit values
            int response = (int)dst_norm.at<float>(j, i);

            // only store points above a required minimum threshold (s. Harris detector parameters)
            if (response > minResponse)
            {
                // create new keypoint from keypoint (corner) detector response at the current image location (i, j)
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);  // keypoint coordinates
                newKeyPoint.size = 2 * apertureSize; // keypoint diameter (region of interest) = 2 * Sobel filter aperture size
                newKeyPoint.response = response;     // keypoint detector response

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                { // loop over all previous keypoints found so far
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it); // get current keypoint overlap
                    if (kptOverlap > maxOverlap)
                    { // if keypoints overlap check which one contains a stronger response and keep the largest
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is > t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                { // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }

        } // eof loop over cols

    } // eof loop over rows

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); // stop timer
    cout << "Harris feature detection with n="
        << keypoints.size()
        << " keypoints in "
        << 1000 * t / 1.0
        << " ms"
        << endl;

    // visualize results
    if (bVis)
    {
        // plot image with keypoints
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(
            img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, cv::WINDOW_NORMAL); // cv::WINDOW_NORMAL = 1 enables you to resize the window);
        imshow(windowName, visImage);
        if (bWait)
        {
            cout << "Press key to continue" << endl;
            cv::waitKey(0); // wait for key to be pressed
            cv::destroyWindow(windowName); // close window
        }
    }

    // return processing time for keypoint detection
    return t;
}


// detect keypoints using different newer feature detectors from OpenCV like FAST, BRISK, ...) ... except 
// for the traditional Shi-Tomasi and Harris detectors, which are implemented separately (s. above)
double detKeypointsModern(
    std::vector<cv::KeyPoint> & keypoints, cv::Mat & img, std::string detectorType, bool bVis, bool bWait)
{    
    // initialize feature detector
    cv::Ptr<cv::FeatureDetector> detector;

    if (detectorType.compare("FAST") == 0)
	{	
        /* FAST feature detector
        *  OpenCV: https://docs.opencv.org/4.1.2/df/d74/classcv_1_1FastFeatureDetector.html
        */

        // set FAST feature detector parameters
        // int threshold=30;
        // int threshold=20;
        int threshold=10; // threshold on difference between intensity of the central pixel and pixels of a circle around this pixel
        bool nonmaxSuppression=true; // if true, non-maximum suppression is applied to detected corners (keypoints)
        cv::FastFeatureDetector::DetectorType type=cv::FastFeatureDetector::TYPE_9_16; // neighborhoods: TYPE_5_8, TYPE_7_12, TYPE_9_16

        // create FAST feature detector
        detector = cv::FastFeatureDetector::create(threshold, nonmaxSuppression, type);
	}
	else if (detectorType.compare("BRISK") == 0)
	{
        /* BRISK (Binary robust invariant scalable keypoints) feature detector and descriptor extractor
        *  OpenCV: https://docs.opencv.org/4.1.2/de/dbf/classcv_1_1BRISK.html
        */

        // set BRISK feature detector parameters
        // int threshold = 60;
        int threshold = 30; // AGAST detection threshold score
		int octaves = 3; // detection octaves. Use 0 to do single scale
		float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint

        // create BRISK feature detector
		detector = cv::BRISK::create(threshold, octaves, patternScale);
	}
	else if (detectorType.compare("ORB") == 0)
	{
        /* ORB (Oriented FAST and Rotated BRIEF) feature detector and descriptor extractor
        *  OpenCV: https://docs.opencv.org/4.1.2/db/d95/classcv_1_1ORB.html
        */

        // set ORB feature detector parameters
        int nfeatures = 500; // maximum number of features to retain
		float scaleFactor = 1.2f; // pyramid decimation ratio, greater than 1 (scaleFactor==2 => classical pyramid)
        int nlevels = 8; // number of pyramid levels (smallest level has linear size equal to input_image_linear_size/pow(scaleFactor, nlevels - firstLevel))
		int edgeThreshold = 31; // size of the border where the features are not detected (should roughly match the patchSize parameter)
		int firstLevel = 0; // level of pyramid to put source image to (Previous layers are filled with upscaled source image)
		int WTA_K = 2; // number of points that produce each element of the oriented BRIEF descriptor (default value: 2, other possible values: 3, 4)
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE; // default: use HARRIS_SCORE to rank features (or: FAST_SCORE => slightly less stable keypoints, but faster)
		int patchSize = 31; // size of the patch used by the oriented BRIEF descriptor
		int fastThreshold = 20; // the fast threshold

        // create ORB feature detector
		detector = cv::ORB::create(
            nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel,
            WTA_K, scoreType, patchSize, fastThreshold);
	}
	else if (detectorType.compare("KAZE") == 0)
	{
        /* KAZE feature detector and descriptor extractor
        *  OpenCV: https://docs.opencv.org/4.1.2/d3/d61/classcv_1_1KAZE.html
        */

        // set KAZE feature detector parameters
        bool extended = false; // set to enable extraction of extended (128-byte) descriptor
		bool upright = false; // set to enable use of upright descriptors (non rotation-invariant)
		float threshold = 0.001f; // detector response threshold to accept point
		int nOctaves = 4; // maximum octave evolution of the image
		int nOctaveLayers = 4; // default number of sublevels per scale level
		cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2; // options: DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER

        // create KAZE feature detector
        detector = cv::KAZE::create(
            extended, upright, threshold, nOctaves, nOctaveLayers, diffusivity); 	
	}
	else if (detectorType.compare("AKAZE") == 0)
	{
        /* AKAZE (Accelerated-KAZE) feature detector and descriptor extractor
        *  OpenCV: https://docs.opencv.org/4.1.2/d8/d30/classcv_1_1AKAZE.html
        */

        // set AKAZE feature detector parameters
        cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB; // options: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT
		int descriptor_size = 0; // size of the descriptor in bits. 0 -> full size
		int descriptor_channels = 3; // number of channels in the descriptor (1, 2, 3)
		float threshold = 0.001f; // detector response threshold to accept point
		int nOctaves = 4; // maximum octave evolution of the image
		int nOctaveLayers = 4; // default number of sublevels per scale level
		cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2; // options: DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER
        
        // create AKAZE feature detector
        detector = cv::AKAZE::create(
            descriptor_type, descriptor_size, descriptor_channels, threshold,
            nOctaves, nOctaveLayers, diffusivity);
	}
	else if (detectorType.compare("SIFT") == 0)
	{
        /* SIFT (Scale Invariant Feature Transform) feature detector and descriptor extractor
        *  https://docs.opencv.org/4.1.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
        */

        // set SIFT feature detector parameters
        int nfeatures = 0; // number of best features to retain (features are ranked by their scores measured as the local contrast)
        int nOctaveLayers = 3; // number of layers in each octave (3 is the value used in D. Lowe paper)
        double contrastThreshold = 0.04; // contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions
        double edgeThreshold = 10; // threshold used to filter out edge-like features
        double sigma = 1.6; // sigma of the Gaussian applied to the input image at the octave #0

        // create SIFT feature detector
		detector = cv::xfeatures2d::SIFT::create(
            nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
	}
    else if (detectorType.compare("SURF") == 0)
    {
        /* SURF (Speeded-up robust features) feature detector and descriptor extractor
        *  OpenCV: https://docs.opencv.org/4.1.2/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html#details
        */

        // set SURF feature detector parameters
        double hessianThreshold = 100; // threshold for hessian keypoint detector used in SURF
		int nOctaves = 4; // number of pyramid octaves the keypoint detector will use
		int nOctaveLayers = 3; // number of octave layers within each octave
		bool extended = false; // extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors)
		bool upright = false; // up-right or rotated features flag (true - do not compute orientation of features; false - compute orientation)

        // create SURF feature detector
        detector = cv::xfeatures2d::SURF::create(
            hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
    }
	else {
        // throw error message
        throw "Error: Wrong input argument to detKeypoints(): Detector type not defined!";
	}

    // detect keypoints (timed process)
	double t = (double)cv::getTickCount(); // trigger timer
	detector->detect(img, keypoints); // Detect keypoints
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency(); // stop timer
    cout << detectorType
        << " feature detection with n="
        << keypoints.size()
        << " keypoints in "
        << 1000 * t / 1.0
        << " ms"
        << endl;

	// visualize results
	if (bVis)
	{
        // plot image with keypoints
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(
            img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		string windowName = detectorType.append(" Detector Results");
		cv::namedWindow(windowName, cv::WINDOW_NORMAL); // cv::WINDOW_NORMAL = 1 enables you to resize the window
		imshow(windowName, visImage);
        if (bWait)
        {
            cout << "Press key to continue" << endl;
            cv::waitKey(0); // wait for key to be pressed
            cv::destroyWindow(windowName); // close window
        }
	}

    // return processing time for keypoint detection
    return t;
}


// compare the strength of the detector response of tow different keypoints for sorting
bool compareKeypointResponse(const cv::KeyPoint & kpt1, const cv::KeyPoint & kpt2)
{
    // return true if response of kpt1 is greater than the response of kpt2, or false otherwise
    return kpt1.response > kpt2.response;
}


// limit number of keypoints sorted by the strength of the detector repsonse (Only experimental => Do NOT in real application!)
void limitKeypoints(std::vector<cv::KeyPoint> & keypoints, cv::Mat & imgGray, std::string detectorType, int maxKpts, bool bVis, bool bWait)
{
    // copy keypoints for debugging and visualization purpose
    vector<cv::KeyPoint> keypoints_cpy;
    copy(keypoints.begin(), keypoints.end(), back_inserter(keypoints_cpy));

    if (detectorType.compare("SHITOMASI") == 0)
    { // there is no response info, so keep the first maxKepyoints as they are sorted in descending quality order
        keypoints.erase(keypoints.begin() + maxKpts, keypoints.end());
    }
    else
    {
        // sort keypoints according to the strength of the detector response (keypoints are not always sorted automatically!)
        sort(keypoints.begin(), keypoints.end(), compareKeypointResponse);

        // keep the first maxKpts from the list sorted by descending order of the detector response
        keypoints.erase(keypoints.begin() + maxKpts, keypoints.end());
    }
    cv::KeyPointsFilter::retainBest(keypoints, maxKpts);
    cout << "NOTE: Keypoints have been limited (n_max = " << maxKpts << ")!" << endl;
    cout << "The first n_max = " << keypoints.size() << " keypoints are kept. " << endl;

    // viszalize keypoints before and after limiting their number
    if (bVis)
    {
        // plot original keypoints (copy) before limiting
        cv::Mat visImgGrayAllKpts = imgGray.clone();
        cv::drawKeypoints(
            imgGray, keypoints_cpy, visImgGrayAllKpts,
            cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        // plot keypoints after limiting
        cv::Mat visImgGrayLimitKpts = imgGray.clone();
        cv::drawKeypoints(
            imgGray, keypoints, visImgGrayLimitKpts,
            cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        // vertically concatenate both plots
        cv::Mat visImgGrayKpts;
        cv::vconcat(visImgGrayAllKpts, visImgGrayLimitKpts, visImgGrayKpts);
        // show concatenated plots
        string windowName = "Keypoints before and after limiting their number";
        cv::namedWindow(windowName, cv::WINDOW_NORMAL); // cv::WINDOW_NORMAL = 1 the user can resize the window (no constraint)                    
        //cv::namedWindow(windowName, cv::WINDOW_FREERATIO); // cv::WINDOW_FREERATIO = 5 the image expends as much as it can (no ratio constraint). 
        cv::imshow(windowName, visImgGrayKpts);
        if (bWait)
        {
            cout << "Press key to continue" << endl;
            cv::waitKey(0); // wait for key to be pressed
            cv::destroyWindow(windowName); // close window
        }  
    }

    // clear the copy of the original keypoint vector
    keypoints_cpy.clear();
}
