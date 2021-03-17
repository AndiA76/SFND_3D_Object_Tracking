
// ============================================================================
//  
//  Project 2.2: 3D Object Tracking (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
//  
//  Source:      https://github.com/udacity/SFND_3D_Object_Tracking
//
// ============================================================================

// function definitions for camera image and Lidar point cloud data fusion

#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

#include "camFusion.hpp"
#include "matching2D.hpp"
#include "lidarData.hpp"
#include "dataStructures.h"

using namespace std;
        

// cluster Lidar points by associating their projections onto image plane to 2D bounding boxes of object detections
void clusterLidarWithROI(
    std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor,
    cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    /* Create groups of Lidar points whose projection into the camera falls into the same bounding box and associate
    *  them with the latter if the association is unique. Shrink each bounding box by a given percentage to avoid 3D 
    *  object merging at the edges of an ROI. Do not associate Lidar points that fall into multiple bounding boxes.
    */
    
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project 3D Lidar point into 2D camera image plane
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // 2D pixel coordinates in image plane
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        // create vector of pointers to all bounding boxes which enclose the current Lidar point
        vector<vector<BoundingBox>::iterator> enclosingBoxes;

        // loop over all 2D object bounding boxes detected in the color image
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been 
* manually tuned to fit the 2000x2000 size. However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(
    std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait,
    std::string windowName)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xw_min=1e8, xw_max=-1e8, yw_min=1e8, yw_max=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from the sensor
            float yw = (*it2).y; // world position in m with y facing left from the sensor
            xw_min = xw_min<xw ? xw_min : xw;
            xw_max = xw_max>xw ? xw_max : xw;
            yw_min = yw_min<yw ? yw_min : yw;
            yw_max = yw_max>yw ? yw_max : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point - scale the size of the Lidar point according to its reflectivity value
            int circleSize = int(5*(*it2).r); // scale circle size by Lidar point reflectivity (r = 0...1)
            cv::circle(topviewImg, cv::Point(x, y), circleSize, currColor, -1);
        }

        // draw enclosing rectangle
        float lineThickness = 1.0;
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0,0,0), lineThickness);

        // augment object with some key data
        float fontSize = 0.75;
        char str1[200], str2[200], str3[200], str4[200], str5[200];
        sprintf(str1, "boxID=%d, #pts=%d (scaled by reflectivity)", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-50, bottom+25), cv::FONT_ITALIC, fontSize, currColor);
        sprintf(str2, "yw_max-yw_min=%2.2f m", yw_max-yw_min);
        putText(topviewImg, str2, cv::Point2f(left-50, bottom+50), cv::FONT_ITALIC, fontSize, currColor);
        sprintf(str3, "xw_min=%2.2f m, xw_max=%2.2f m", xw_min, xw_max);
        putText(topviewImg, str3, cv::Point2f(left-50, bottom+75), cv::FONT_ITALIC, fontSize, currColor);

        // get the mean and median value of the x world coordinates of all Lidar points on the target object
        int N = it1->lidarPoints.size();
        if (N > 0)
        { // only the bounding box of the target object has associated Lidar points
            vector<float> xw_lidarPoints(N);
            for (int i = 0; i < N; ++i)
            {
                xw_lidarPoints[i] = it1->lidarPoints[i].x; // x world coordinate of current lidar Point in m
            }
            // calculate mean and standard deviation of the x wolrd coordinates of all Lidar points
            pair<float, float> out = calcMeanAndStandardDeviation(xw_lidarPoints); // mean and standard deviation of the x world coordinates
            float xw_mean = out.first;
            float xw_std = out.second;
            // get median of the x wolrd coordinates of all Lidar points
            float xw_median = median(xw_lidarPoints);

            // transform world to image coordinates
            int y_mean = (-xw_mean * imageSize.height / worldSize.height) + imageSize.height;
            int y_median = (-xw_median * imageSize.height / worldSize.height) + imageSize.height;
            
            // draw horizontal lines for mean (blue) x-value and median (green) x-value
            cv::line(topviewImg, cv::Point2f(left-50, y_mean), cv::Point2f(right+50, y_mean), cv::Scalar(200, 0, 0), lineThickness, cv::LINE_AA);
            cv::line(topviewImg, cv::Point2f(left-50, y_median), cv::Point2f(right+50, y_median), cv::Scalar(0, 200, 0), lineThickness, cv::LINE_AA);

            // add mean & standard deviation and median over all target world positions in m with x facing forward from the sensor
            sprintf(str3, "xw_mean=%2.2f m, xw_std=%2.2f m", xw_mean, xw_std);
            putText(topviewImg, str3, cv::Point2f(left-50, bottom+100), cv::FONT_ITALIC, fontSize, cv::Scalar(200, 0, 0));
            sprintf(str4, "xw_median=%2.2f m", xw_median);
            putText(topviewImg, str4, cv::Point2f(left-50, bottom+125), cv::FONT_ITALIC, fontSize, cv::Scalar(0, 200, 0));
        }
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    cv::namedWindow(windowName, cv::WINDOW_NORMAL); // cv::WINDOW_NORMAL = 1 enables to resize the window
    //cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE); // cv::WINDOW_AUTOSIZE = 2 adjusts automatically to fit the size of the image
    cv::imshow(windowName, topviewImg);
    if (bWait)
    {
        cout << "Press key to continue" << endl;
        cv::waitKey(0); // wait for key to be pressed
        cv::destroyWindow(windowName); // close window
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(
    BoundingBox &prevBB, BoundingBox &currBB, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
    std::vector<cv::DMatch> &kptMatches)
{
    // loop over all keypoint matches
    for ( cv::DMatch &kptMatch : kptMatches )
    {
        // check if the keypoint match from the current frame lies within the ROI of the current bounding box
        if ( currBB.roi.contains(kptsCurr[kptMatch.trainIdx].pt) )
        {
            /* double check if the associated keypoint match from the previous frame lies within the ROI of the 
            *  previous bounding box in order to filter out false positive keypoint matches outside the ROI */
            if ( prevBB.roi.contains(kptsPrev[kptMatch.queryIdx].pt) )
            {
                // Save matching keypoints associated with the current bounding box
                currBB.keypoints.push_back(kptsCurr[kptMatch.trainIdx]);

                // Save keypoint matches associated with the current bounding box
                currBB.kptMatches.push_back(kptMatch);
            }
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(
    BoundingBox &prevBB, BoundingBox &currBB, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
    std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg, bool bVis, bool bWait, 
    bool bPrintDebugInfo)
{
    /* computeTTCCamera() estimates the time-to-collision using the median distance ratio between keypoints associated
    *  with a given bounding box around a target object in two subsequent image frames. The distances between the same
    *  matching keypoints are calculated for the keypoint matches in the target bounding box of the current frame 
    *  (= reference or train image) as well as for the keypoint matches in the target bounding box of the previous frame
    *  (= source or query image). The distance ratio is only taken if the distance is larger than a minimum threshold 
    *  in order to prevent for distortion effects for too small values or to filter out distances between a keypoint and
    *  itself, which whould lead to zero distances.
    * 
    *  prerequisite: keypoint matches are associated to the bounding boxes prior to calling this function
    */

    // ratios of mutual distances between keypoints of the previous / current frame associated with the given bounding box
    vector<double> distRatios;

    // minimum threshold (at least numerical minimum) to filter out distances between a keypoint and itself or too low values
    double minDistThreshold = min( 
        min(prevBB.roi.width, prevBB.roi.height),
        min(currBB.roi.width, currBB.roi.height)
        ) / 2;
    double minDistCurr = max(abs(minDistThreshold), numeric_limits<double>::min());
    double minDistPrev = max(abs(minDistThreshold), numeric_limits<double>::min());

    // outer loop to pick a first keypoint from previous and current frame
    for ( auto it1 = kptMatches.begin(); it1 != kptMatches.end(); it1++ )
    {
        /* check if the matching keypoint pair from the current frame and the previous frame
        *  lies within the ROI of the current bounding box and the previous bounding box */
        if ( currBB.roi.contains(kptsCurr[it1->trainIdx].pt) 
            && prevBB.roi.contains(kptsPrev[it1->queryIdx].pt) )
        {
            // get current keypoint and its matched partner in the prev. frame
            cv::KeyPoint kptCurr1 = kptsCurr.at(it1->trainIdx);
            cv::KeyPoint kptPrev1 = kptsPrev.at(it1->queryIdx);

            if (bPrintDebugInfo)
            {
                cout << "Mutual distances between keypoints matches in the current ROI and the previous ROI:" << endl;
            }
            
            // inner loop to pick a second keypoint from previous and current frame
            for ( auto it2 = kptMatches.begin(); it2 != kptMatches.end(); it2++ )
            {
                /* check if the matching keypoint pair from the current frame and the previous frame
                *  lies within the ROI of the current bounding box and the previous bounding box */
                if ( currBB.roi.contains(kptsCurr[it2->trainIdx].pt) 
                    && prevBB.roi.contains(kptsPrev[it2->queryIdx].pt) )
                {
                    // get next keypoint and its matched partner in the prev. frame
                    cv::KeyPoint kptCurr2 = kptsCurr.at(it2->trainIdx);
                    cv::KeyPoint kptPrev2 = kptsPrev.at(it2->queryIdx);

                    // calculate the mutual distances between two keypoints of the previous frame and the current frame
                    double distCurr = cv::norm(kptCurr1.pt - kptCurr2.pt);
                    double distPrev = cv::norm(kptPrev1.pt - kptPrev2.pt);

                    if (bPrintDebugInfo)
                    {
                        cout << "distCurr = " << distCurr << " [pixel] " << endl;
                        cout << "distPrev = " << distPrev << " [pixel] " << endl;
                    }

                    // Drop any distances from the current / previous frame smaller or equal than a minmimum threshold
                    if ( ( abs(distCurr) > minDistCurr ) && ( abs(distPrev) > minDistPrev ) )
                    {
                        if (bPrintDebugInfo)
                        {
                            cout << "distCurr = " << distCurr << " [pixel] (considered) " << endl;
                            cout << "distPrev = " << distPrev << " [pixel] (considered) " << endl;
                        }

                        // save current distance ratio (distance ratios become larger for approaching targets)
                        // distance ratio > 1 if target approaches | distance ratio < 1 if target recedes
                        distRatios.push_back( distCurr / distPrev );
                    }
                }
            }
        }
    }

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }
    else
    {
        // pick the median distance ratio to reduce the influence of outliers
        double medianDistRatio = median(distRatios);

        // get current sample time between two consecutive frames
        double dT = 1 / frameRate;

        // estimate time to collision (TTC) assuming a constant velocity model
        // (TTC > 0 if distance to the target shrinks | TTC < 0 if distance to the target increases)
        TTC = -dT / (1 - medianDistRatio);

        // print TTC estimation results
        cout << "Time-to-collision (TTC) estimation using Camera:" << endl;
        cout << "computeTTCCamera => using median keypoint distance ratio" << endl;
        cout << "n_KptMatches = " << kptMatches.size() << endl;
        cout << "n_DistRatios = " << distRatios.size() << endl;
        cout << "medianDistRatio = " << medianDistRatio << endl;
        cout << "dT = " << dT << endl;
        cout << "TTC = " << TTC << endl;
    }

    if ( bVis && ( visImg != nullptr) )
    {
        // show 2D keypoint matches associated with the target bounding box in current image frame
        string windowName = "2D keypoint matches associated with the target bounding box (current frame)";
        cv::Mat currImg = (*visImg).clone();
        cv::drawKeypoints(*visImg, currBB.keypoints, currImg,
             cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        // add target object bounding box (= ROI) in green
        cv::rectangle(currImg, cv::Point(currBB.roi.x, currBB.roi.y), 
            cv::Point(currBB.roi.x + currBB.roi.width, currBB.roi.y + currBB.roi.height),
            cv::Scalar(0, 255, 0), 2);
        // add time-to-collision calculation results to the image
        char text_str[200];
        sprintf(text_str, "TTC Camera : %f s", TTC);
        putText(currImg, text_str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));
        cv::namedWindow(windowName, cv::WINDOW_NORMAL); // cv::WINDOW_NORMAL = 1 the user can resize the window (no constraint) 
        cv::imshow(windowName, currImg);
        if (bWait)
        {
            cout << "Press key to continue" << endl;
            cv::waitKey(0); // wait for key to be pressed
            cv::destroyWindow(windowName); // close window
        }
    }
}


// estimate time to collision from Lidar points associated with an target object bounding box
void computeTTCLidar(
    std::vector<LidarPoint> &lidarPointsPrev, std::vector<LidarPoint> &lidarPointsCurr,
    double frameRate, double &TTC, int option,
    BoundingBox &currBB, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT,
    cv::Mat *visImg, bool bVis, bool bWait)
{
    /* options how to estimate a (robust) distance estimate from the Lidar points in the ROI
    *  1: closest distance
    *  2: mean distance
    *  3: closest distance larger than mean distance minus x-times (x = 1.0) standard deviation {x = 0.0 ... 3.0 or 4.0}
    *  4: median distance
    *  5: threshold (= 0.33) on sorted distances {threshold = 0: closest point | 0.5: median | 1.0: farthest point} 
    */

    // estimate time to collision using selected distance estimation method option
    if ( option == 5 )
    {
        // threshold on sorted distances
        cout << "compute TTCLidar => option = 5: threshold on sorted distances" << endl;

        // threshold (0...1) to pick a reference measurement from all Lidar distances sorted in ascending order
        float threshold = 0.33; // threshold = 0: closest point | 0.5: median | 1.0: farthest point

        // get the number of Lidar points on the target object in the previous frame and in the current frame
        int n_lidarPointsPrev = lidarPointsPrev.size();
        int n_lidarPointsCurr = lidarPointsCurr.size();

        // check if there are Lidar points on the target object in the previous frame and in the current frame
        if ( (n_lidarPointsPrev == 0) || (n_lidarPointsCurr == 0) )
        {
            TTC = NAN;
            return;
        }
        else
        {
            // get all Lidar distance measurements to the target object in the previous frame
            vector<double> lidarDistPrev(n_lidarPointsPrev);
            for ( int i = 0; i < n_lidarPointsPrev; ++i )
            {
                lidarDistPrev[i] = lidarPointsPrev[i].x;
            }
            // sort the Lidar distances to the target object in the previous frame in ascending order
            sort(lidarDistPrev.begin(), lidarDistPrev.end());
            // get reference index
            int refIdxPrev = floor(threshold * n_lidarPointsPrev);
            // pick the reference distance to the target object in the previous frame
            double refDistPrev = lidarDistPrev[refIdxPrev];

            // get all Lidar distance measurements to the target object in the current frame
            vector<double> lidarDistCurr(n_lidarPointsCurr);
            for ( int i = 0; i < n_lidarPointsCurr; ++i )
            {
                lidarDistCurr[i] = lidarPointsCurr[i].x;
            }
            // sort the Lidar distances to the target object in the current frame in ascending order
            sort(lidarDistCurr.begin(), lidarDistCurr.end());
            // get reference index
            int refIdxCurr = floor(threshold * n_lidarPointsPrev);
            // pick the reference distance to the target object in the current frame
            double refDistCurr = lidarDistCurr[refIdxCurr];

            // calculate the delta distance from two consecutive Lidar measurements to an object of interest
            // dX < 0 when approaching the target | dX > 0 when target recedes
            double dX = refDistCurr - refDistPrev;

            // get the elapsed time difference between the two consecutive Lidar measurments from inverse frame rate
            double dT = 1 / frameRate;

            // estimate the relative velocity assuming a constant velocity model
            // v_rel > 0 if distance to the target shrinks | v_rel < 0 if distance to the target increases
            double v_rel = -dX / dT;

            // estimate time to collision (TTC) assuming a constant velocity model
            // (TTC > 0 if distance to the target shrinks | TTC < 0 if distance to the target increases)
            TTC = refDistCurr / v_rel;

            // print TTC estimation results
            cout << "Time-to-collision (TTC) estimation using Lidar:" << endl;
            cout << "n_lidarPointsCurr = " << n_lidarPointsCurr << endl;
            cout << "n_lidarPointsPrev = " << n_lidarPointsPrev << endl;
            cout << "refIdxCurr = " << refIdxCurr << endl;
            cout << "refIdxPrev = " << refIdxPrev << endl;
            cout << "refDistCurr = " << refDistCurr << endl;
            cout << "refDistPrev = " << refDistPrev << endl;
            cout << "dX = " << dX << endl;
            cout << "dT = " << dT << endl;
            cout << "v_rel = " << v_rel << endl;
            cout << "TTC = " << TTC << endl;
        }
    }
    else if ( option == 4 )
    {
        // median distance
        cout << "compute TTCLidar => option = 4: median distance" << endl;

        // get the number of Lidar points on the target object in the previous frame and in the current frame
        int n_lidarPointsPrev = lidarPointsPrev.size();
        int n_lidarPointsCurr = lidarPointsCurr.size();

        // check if there are Lidar points on the target object in the previous frame and in the current frame
        if ( (n_lidarPointsPrev == 0) || (n_lidarPointsCurr == 0) )
        {
            TTC = NAN;
            return;
        }
        else
        {
            // get all Lidar distance measurements to the target object in the previous frame
            vector<double> lidarDistPrev(n_lidarPointsPrev);
            for ( int i = 0; i < n_lidarPointsPrev; ++i )
            {
                lidarDistPrev[i] = lidarPointsPrev[i].x;
            }
            // pick the median distance to the target object in the previous frame to reduce the influence of outliers
            double medianDistPrev = median(lidarDistPrev);

            // get all Lidar distance measurements to the target object in the current frame
            vector<double> lidarDistCurr(n_lidarPointsCurr);
            for ( int i = 0; i < n_lidarPointsCurr; ++i )
            {
                lidarDistCurr[i] = lidarPointsCurr[i].x;
            }
            // pick the median distance to the target object in the current frame to reduce the influence of outliers
            double medianDistCurr = median(lidarDistCurr);

            // calculate the delta distance from two consecutive Lidar measurements to an object of interest
            // dX < 0 when approaching the target | dX > 0 when target recedes
            double dX = medianDistCurr - medianDistPrev;

            // get the elapsed time difference between the two consecutive Lidar measurments from inverse frame rate
            double dT = 1 / frameRate;

            // estimate the relative velocity assuming a constant velocity model
            // v_rel > 0 if distance to the target shrinks | v_rel < 0 if distance to the target increases
            double v_rel = -dX / dT;

            // estimate time to collision (TTC) assuming a constant velocity model
            // (TTC > 0 if distance to the target shrinks | TTC < 0 if distance to the target increases)
            TTC = medianDistCurr / v_rel;

            // print TTC estimation results
            cout << "Time-to-collision (TTC) estimation using Lidar:" << endl;
            cout << "n_lidarPointsCurr = " << n_lidarPointsCurr << endl;
            cout << "n_lidarPointsPrev = " << n_lidarPointsPrev << endl;
            cout << "medianDistCurr = " << medianDistCurr << endl;
            cout << "medianDistPrev = " << medianDistPrev << endl;
            cout << "dX = " << dX << endl;
            cout << "dT = " << dT << endl;
            cout << "v_rel = " << v_rel << endl;
            cout << "TTC = " << TTC << endl;
        }
    }
    else if ( option == 3 )
    {
        // clostest distance larger than mean distance minus multiplier-* standard deviation
        cout << "compute TTCLidar => option = 3: clostest distance larger than mean distance minus multiplier * standard deviation" << endl;

        // multiplier on standard deviation
        float multiplier = 1.0; // 0 <= x <= 3...4

        // get the number of Lidar points on the target object in the previous frame and in the current frame
        int n_lidarPointsPrev = lidarPointsPrev.size();
        int n_lidarPointsCurr = lidarPointsCurr.size();

        // check if there are Lidar points on the target object in the previous frame and in the current frame
        if ( (n_lidarPointsPrev == 0) || (n_lidarPointsCurr == 0) )
        {
            TTC = NAN;
            return;
        }
        else
        {
            // get all Lidar distance measurements to the target object in the previous frame
            vector<double> lidarDistPrev(n_lidarPointsPrev);
            for ( int i = 0; i < n_lidarPointsPrev; ++i )
            {
                lidarDistPrev[i] = lidarPointsPrev[i].x;
            }
            // calculate mean and standard deviation of all Lidar distances to the target object in the previous frame
            pair<double, double> outPrev = calcMeanAndStandardDeviation(lidarDistPrev);
            double meanDistPrev = outPrev.first;
            double sigmaDistPrev = outPrev.second;
            // find closest distance to the target in previous frame that is still larger than mean distance - multiplier * standard deviation
            double refXPrev = 1e9;
            for ( int i = 0; i < n_lidarPointsPrev; ++i )
            {
                if ( (refXPrev > lidarPointsPrev[i].x) && (lidarPointsPrev[i].x >= (meanDistPrev - multiplier * sigmaDistPrev)) )
                {
                    refXPrev = lidarPointsPrev[i].x;
                }
            }

            // get all Lidar distance measurements to the target object in the current frame
            vector<double> lidarDistCurr(n_lidarPointsCurr);
            for ( int i = 0; i < n_lidarPointsCurr; ++i )
            {
                lidarDistCurr[i] = lidarPointsCurr[i].x;
            }
            // calculate mean and standard deviation of all Lidar distances to the target object in the current frame
            pair<double, double> outCurr = calcMeanAndStandardDeviation(lidarDistCurr);
            double meanDistCurr = outCurr.first;
            double sigmaDistCurr = outCurr.second;
            // find closest distance to the target in current frame that is still larger than mean distance - multiplier * standard deviation
            double refXCurr = 1e9;
            for ( int i = 0; i < n_lidarPointsCurr; ++i )
            {
                if ( (refXCurr > lidarPointsCurr[i].x) && (lidarPointsCurr[i].x >= (meanDistCurr - multiplier * sigmaDistCurr)) )
                {
                    refXCurr = lidarPointsCurr[i].x;
                }
            }

            // calculate the delta distance from two consecutive Lidar measurements to an object of interest
            // dX < 0 when approaching the target | dX > 0 when target recedes
            double dX = refXCurr - refXPrev;

            // get the elapsed time difference between the two consecutive Lidar measurments from inverse frame rate
            double dT = 1 / frameRate;

            // estimate the relative velocity assuming a constant velocity model
            // v_rel > 0 if distance to the target shrinks | v_rel < 0 if distance to the target increases
            double v_rel = -dX / dT;

            // estimate time to collision (TTC) assuming a constant velocity model
            // (TTC > 0 if distance to the target shrinks | TTC < 0 if distance to the target increases)
            TTC = refXCurr / v_rel;

            // print TTC estimation results
            cout << "Time-to-collision (TTC) estimation using Lidar:" << endl;
            cout << "n_lidarPointsCurr = " << n_lidarPointsCurr << endl;
            cout << "n_lidarPointsPrev = " << n_lidarPointsPrev << endl;
            cout << "meanDistCurr = " << meanDistCurr << endl;
            cout << "meanDistPrev = " << meanDistPrev << endl;
            cout << "multiplier * sigmaDistCurr = " << multiplier * sigmaDistCurr << endl;
            cout << "multiplier * sigmaDistPrev = " << multiplier * sigmaDistPrev << endl;
            cout << "meanDistCurr - multiplier * sigmaDistCurr = " << meanDistCurr - multiplier * sigmaDistCurr << endl;
            cout << "meanDistPrev - multiplier * sigmaDistPrev = " << meanDistPrev - multiplier * sigmaDistPrev << endl;
            cout << "refXCurr = " << refXCurr << endl;
            cout << "refXPrev = " << refXPrev << endl;
            cout << "dX = " << dX << endl;
            cout << "dT = " << dT << endl;
            cout << "v_rel = " << v_rel << endl;
            cout << "TTC = " << TTC << endl;
        }
    }
    else if ( option == 2 )
    {
        // mean distance
        cout << "compute TTCLidar => option = 2: mean distance" << endl;

        // get the number of Lidar points on the target object in the previous frame and in the current frame
        int n_lidarPointsPrev = lidarPointsPrev.size();
        int n_lidarPointsCurr = lidarPointsCurr.size();

        // check if there are Lidar points on the target object in the previous frame and in the current frame
        if ( (n_lidarPointsPrev == 0) || (n_lidarPointsCurr == 0) )
        {
            TTC = NAN;
            return;
        }
        else
        {
            // get all Lidar distance measurements to the target object in the previous frame
            vector<double> lidarDistPrev(n_lidarPointsPrev);
            for ( int i = 0; i < n_lidarPointsPrev; ++i )
            {
                lidarDistPrev[i] = lidarPointsPrev[i].x;
            }
            // calculate mean of all Lidar distances to the target object in the previous frame
            double meanDistPrev = mean(lidarDistPrev);

            // get all Lidar distance measurements to the target object in the current frame
            vector<double> lidarDistCurr(n_lidarPointsCurr);
            for ( int i = 0; i < n_lidarPointsCurr; ++i )
            {
                lidarDistCurr[i] = lidarPointsCurr[i].x;
            }
            // calculate mean of all Lidar distances to the target object in the current frame
            double meanDistCurr = mean(lidarDistCurr);

            // calculate the delta distance from two consecutive Lidar measurements to an object of interest
            // dX < 0 when approaching the target | dX > 0 when target recedes
            double dX = meanDistCurr - meanDistPrev;

            // get the elapsed time difference between the two consecutive Lidar measurments from inverse frame rate
            double dT = 1 / frameRate;

            // estimate the relative velocity assuming a constant velocity model
            // v_rel > 0 if distance to the target shrinks | v_rel < 0 if distance to the target increases
            double v_rel = -dX / dT;

            // estimate time to collision (TTC) assuming a constant velocity model
            // (TTC > 0 if distance to the target shrinks | TTC < 0 if distance to the target increases)
            TTC = meanDistCurr / v_rel;

            // print TTC estimation results
            cout << "Time-to-collision (TTC) estimation using Lidar:" << endl;
            cout << "n_lidarPointsCurr = " << n_lidarPointsCurr << endl;
            cout << "n_lidarPointsPrev = " << n_lidarPointsPrev << endl;
            cout << "meanDistCurr = " << meanDistCurr << endl;
            cout << "meanDistPrev = " << meanDistPrev << endl;
            cout << "dX = " << dX << endl;
            cout << "dT = " << dT << endl;
            cout << "v_rel = " << v_rel << endl;
            cout << "TTC = " << TTC << endl;
        }
    }
    else
    {
        // closest distance
        cout << "compute TTCLidar => option = 1: closest distance" << endl;

        // get the number of Lidar points on the target object in the previous frame and in the current frame
        int n_lidarPointsPrev = lidarPointsPrev.size();
        int n_lidarPointsCurr = lidarPointsCurr.size();

        // calculate the Lidar distance measurement to the object of interest in the previous frame
        double minXPrev = 1e9;
        for( auto iter=lidarPointsPrev.begin(); iter!=lidarPointsPrev.end(); ++iter ) { // find closest Lidar point
            minXPrev = minXPrev>iter->x ? iter->x : minXPrev;
        }

        // calculate the Lidar distance measurement to the object of interest in the current frame
        double minXCurr = 1e9;
        for( auto iter=lidarPointsCurr.begin(); iter!=lidarPointsCurr.end(); ++iter ) { // find closest Lidar point
            minXCurr = minXCurr>iter->x ? iter->x : minXCurr;
        }

        // calculate the delta distance from two consecutive Lidar measurements to an object of interest
        // dX < 0 when approaching the target | dX > 0 when target recedes
        double dX = minXCurr - minXPrev;

        // get the elapsed time difference between the two consecutive Lidar measurments from inverse frame rate
        double dT = 1 / frameRate;

        // estimate the relative velocity assuming a constant velocity model
        // v_rel > 0 if distance to the target shrinks | v_rel < 0 if distance to the target increases 
        double v_rel = -dX / dT;

        // estimate time to collision (TTC) assuming a constant velocity model
        // (TTC > 0 if distance to the target shrinks | TTC < 0 if distance to the target increases)
        TTC = minXCurr / v_rel;

        // print TTC estimation results
        cout << "Time-to-collision (TTC) estimation using Lidar:" << endl;
        cout << "n_lidarPointsCurr = " << n_lidarPointsCurr << endl;
        cout << "n_lidarPointsPrev = " << n_lidarPointsPrev << endl;
        cout << "minXCurr = " << minXCurr << endl;
        cout << "minXPrev = " << minXPrev << endl;
        cout << "dX = " << dX << endl;
        cout << "dT = " << dT << endl;
        cout << "v_rel = " << v_rel << endl;
        cout << "TTC = " << TTC << endl;
    }

    if ( bVis && ( visImg != nullptr) )
    {
        // show Lidar points projected into the target bounding box on 2D image plane in the current frame
        string windowName = "3D LiDAR points associated with the target bounding box projected on 2D image plane (current frame)";
        cv::Mat currImg = (*visImg).clone();
        // overlay a projection of the 3D Lidar points onto the image plane
        showLidarImgOverlay(currImg, currBB.lidarPoints, P_rect_xx, R_rect_xx, RT, &currImg, false, windowName);
        // add target object bounding box (= ROI) in green
        cv::rectangle(currImg, cv::Point(currBB.roi.x, currBB.roi.y), 
            cv::Point(currBB.roi.x + currBB.roi.width, currBB.roi.y + currBB.roi.height),
            cv::Scalar(0, 255, 0), 2);
        // add time-to-collision calculation results to the image
        char text_str[200];
        sprintf(text_str, "TTC Lidar : %f s", TTC);
        putText(currImg, text_str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));
        cv::namedWindow(windowName, cv::WINDOW_NORMAL); // cv::WINDOW_NORMAL = 1 the user can resize the window (no constraint) 
        cv::imshow(windowName, currImg);
        if (bWait)
        {
            cout << "Press key to continue" << endl;
            cv::waitKey(0); // wait for key to be pressed
            cv::destroyWindow(windowName); // close window
        }
    }
}


// track target object bounding boxes between consecutive frames using keypoint matches associated with the bounding boxes
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches,
                        DataFrame &prevFrame, DataFrame &currFrame)
{
    // Remark: std::map is a sorted associative container that contains key-value pairs with unique keys.

    // define an inner map with the boxIDs from the current frame that have common keypoint matches with a given
    // bounding box from the previous frame as key, and a counter for how many key point matches exist as value
    typedef map<int, int> innerMap;

    // define an outer map with the boxIDs from the previous frame that have common keypoint matches with bounding
    // boxes from the current frame as key, and an innerMap (s. above) as value
    typedef map<int, innerMap> outerMap;

    // create a nested map, where the inner map contains a counter as value
    outerMap bbMatchIdMap;

    // loop over all keypoint descriptor matches between the current and the previous frame
    for ( auto iterKptMatch = matches.begin(); iterKptMatch != matches.end(); iterKptMatch++ )
    {
        // loop over all bounding boxes in the previous frame and ...
        for ( auto prevFrameBB = prevFrame.boundingBoxes.begin(); prevFrameBB != prevFrame.boundingBoxes.end(); prevFrameBB++ )
        {
            // ... check if the keypoint match is inside the actual bounding box of the previous frame ...
            if ( prevFrameBB->roi.contains(prevFrame.keypoints.at(iterKptMatch->queryIdx).pt) )
            {
                // ... if yes, loop over all bounding boxes in the current frame and ...
                for ( auto currFrameBB = currFrame.boundingBoxes.begin(); currFrameBB != currFrame.boundingBoxes.end(); currFrameBB++ )
                {
                    // ... check if the keypoint match is inside the actual bounding box of the current frame ...
                    if ( currFrameBB->roi.contains(currFrame.keypoints.at(iterKptMatch->trainIdx).pt) )
                    {
                        /* ... if yes we have found a bounding box pair with a keypoint match => store the pair of boxIDs
                        
                        Remark: The same pair of boxIDs will appear as many times in the nested map as there are keypoint 
                        matches between the previous image and the current image within these two bounding boxes.
                        */
                        if ( bbMatchIdMap.find(prevFrameBB->boxID) == bbMatchIdMap.end() )
                        {
                            // Outer key not found => insert new pair of bounding box matches and set counter to 1
                            innerMap tempInnerMap;
                            tempInnerMap.insert(make_pair(currFrameBB->boxID, 1));
                            bbMatchIdMap.insert(make_pair(prevFrameBB->boxID, tempInnerMap));
                        }
                        else
                        {
                            // Outer key found => check for inner key
                            if ( bbMatchIdMap[prevFrameBB->boxID].find(currFrameBB->boxID) == bbMatchIdMap[prevFrameBB->boxID].end() )
                            {
                                // Inner key not found => insert new pair of bounding box matches and set counter to 1
                                bbMatchIdMap[prevFrameBB->boxID].insert(make_pair(currFrameBB->boxID, 1));
                            }
                            else
                            {
                                // Outer and inner key found => element already exists => increment counter
                                bbMatchIdMap[prevFrameBB->boxID][currFrameBB->boxID]++;
                            }
                        }        
                    }
                }
            }
        }
    }

    // loop over the nested map of bounding box matches from the current and the previous frame and ...
    // select the bounding box pair with most keypoint matches
    outerMap::iterator outerMapIterator;
    innerMap::iterator innerMapIterator;
    for ( outerMapIterator=bbMatchIdMap.begin(); outerMapIterator!=bbMatchIdMap.end(); outerMapIterator++ )
    {
        int maxCount = 0;
        int maxCountBoxID = 0;
        for ( innerMapIterator=(outerMapIterator->second).begin(); innerMapIterator!=(outerMapIterator->second).end(); innerMapIterator++ )
        {
            if ( innerMapIterator->second > maxCount )
            {
                maxCountBoxID = innerMapIterator->first;
                maxCount = innerMapIterator->second;
            }
            /* debugging information => comment */
            cout << "prevFrameBB->boxID = " << outerMapIterator->first << "\t"
                << "currFrameBB->boxID = " << innerMapIterator->first << "\t"
                << "frequency = " << innerMapIterator->second << endl;
        }
        /* debugging information => comment */
        cout << "prevFrameBB->boxID = " << outerMapIterator->first << "\t"
            << "currFrameBB->boxID = " << maxCountBoxID << "\t"
            << "frequency = " << maxCount << "(selected)" << endl;

        // Store the bounding box pairs from the current and the previous frame with the most key point matches
        bbBestMatches.insert(make_pair(outerMapIterator->first, maxCountBoxID));
    }    
}
