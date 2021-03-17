// ============================================================================
//  
//  Project 2.2: 3D Object Tracking (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
//  
//  Source:      https://github.com/udacity/SFND_3D_Object_Tracking
//
// ============================================================================

// function definitions for Lidar point cloud data processing and visualization

#include <iostream>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "lidarData.hpp"


using namespace std;

// remove Lidar points based on min. and max distance in X, Y and Z
void cropLidarPoints(std::vector<LidarPoint> &lidarPoints,
    float minX, float maxX, float maxY, float minZ, float maxZ, float minR)
{
    std::vector<LidarPoint> newLidarPts;
    // Loop over all Lidar points
    for(auto it=lidarPoints.begin(); it!=lidarPoints.end(); ++it)
    {
        // Check if current Lidar point is inside the defined ROI boundaries ...
        if( (*it).x>=minX && (*it).x<=maxX
            && (*it).z>=minZ && (*it).z<=maxZ && (*it).z<=0.0
            && abs((*it).y)<=maxY && (*it).r>=minR )
        {
            // ... if yes then crop it
            newLidarPts.push_back(*it);
        }
    }

    lidarPoints = newLidarPts;
}


// load Lidar points from a given location and store them in a vector
void loadLidarFromFile(vector<LidarPoint> &lidarPoints, string filename)
{
    // allocate 4 MB buffer (only ~130*4*4 KB are needed)
    unsigned long num = 1000000;
    float *data = (float*)malloc(num*sizeof(float));
    
    // pointers
    float *px = data+0;
    float *py = data+1;
    float *pz = data+2;
    float *pr = data+3;
    
    // load point cloud
    FILE *stream;
    stream = fopen (filename.c_str(),"rb");
    num = fread(data,sizeof(float),num,stream)/4;
 
    for (int32_t i=0; i<num; i++) {
        LidarPoint lpt;
        lpt.x = *px; lpt.y = *py; lpt.z = *pz; lpt.r = *pr;
        lidarPoints.push_back(lpt);
        px+=4; py+=4; pz+=4; pr+=4;
    }
    fclose(stream);
}


// plot Lidar points from a bird's eye perspective (top view)
void showLidarTopview(std::vector<LidarPoint> &lidarPoints, cv::Size worldSize, cv::Size imageSize, 
    bool bWait, std::string windowName)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(0, 0, 0));

    // plot Lidar points into image
    for (auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it)
    {
        float xw = (*it).x; // world position in m with x facing forward from sensor
        float yw = (*it).y; // world position in m with y facing left from sensor

        int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
        int x = (-yw * imageSize.height / worldSize.height) + imageSize.width / 2;

        // add distance-dependent colorization
        float zw_roadLevel = -1.40; // z world position of road level 
        float zw = (*it).z; // world position in m with z facing upwards from sensor
        if(zw > zw_roadLevel)
        { // only consider Lidar points above the road level
            float val = it->x;
            float maxVal = worldSize.height;
            int red = min(255, (int)(255 * abs((val - maxVal) / maxVal)));
            int green = min(255, (int)(255 * (1 - abs((val - maxVal) / maxVal))));
            cv::circle(topviewImg, cv::Point(x, y), 3, cv::Scalar(0, green, red), -1);
        }
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y),
                 cv::Scalar(255, 0, 0));
    }

    // display image
    // string windowName = "Top-View Perspective of LiDAR data";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL); // cv::WINDOW_NORMAL = 1 the user can resize the window (no constraint) 
    cv::imshow(windowName, topviewImg);
    if (bWait)
    {
        cout << "Press key to continue" << endl;
        cv::waitKey(0); // wait for key to be pressed
        cv::destroyWindow(windowName); // close window
    }
}


// show image with overlayed Lidar points projected into the target bounding box
void showLidarImgOverlay(cv::Mat &img, std::vector<LidarPoint> &lidarPoints,
    cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT, cv::Mat *extVisImg, bool bWait, std::string windowName)
{
    // init image for visualization
    cv::Mat visImg; 
    if(extVisImg==nullptr)
    {
        visImg = img.clone();
    } else 
    {
        visImg = *extVisImg;
    }

    cv::Mat overlay = visImg.clone();

    // find max. x-value
    double maxVal = 0.0; 
    for(auto it=lidarPoints.begin(); it!=lidarPoints.end(); ++it)
    {
        maxVal = maxVal<it->x ? it->x : maxVal;
    }

    cv::Mat X(4,1,cv::DataType<double>::type);
    cv::Mat Y(3,1,cv::DataType<double>::type);
    for(auto it=lidarPoints.begin(); it!=lidarPoints.end(); ++it) {

            X.at<double>(0, 0) = it->x;
            X.at<double>(1, 0) = it->y;
            X.at<double>(2, 0) = it->z;
            X.at<double>(3, 0) = 1;

            Y = P_rect_xx * R_rect_xx * RT * X;
            cv::Point pt;

            pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
            pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

            float val = it->x;
            int red = min(255, (int)(255 * abs((val - maxVal) / maxVal)));
            int green = min(255, (int)(255 * (1 - abs((val - maxVal) / maxVal))));
            cv::circle(overlay, pt, 5, cv::Scalar(0, green, red), -1);
    }

    // alpha-blending of overlay image (alpha = opacity) on visImg (beta = (1-alpha))
    float opacity = 0.6;
    cv::addWeighted(overlay, opacity, visImg, 1 - opacity, 0, visImg);

    // return augmented image or wait if no image has been provided
    if (extVisImg == nullptr)
    {
        //string windowName = "LiDAR data on image overlay";
        cv::namedWindow( windowName, cv::WINDOW_NORMAL); // cv::WINDOW_NORMAL = 1 the user can resize the window (no constraint) 
        cv::imshow( windowName, visImg );
        if (bWait)
        {
            cout << "Press key to continue" << endl;
            cv::waitKey(0); // wait for key to be pressed
            cv::destroyWindow(windowName); // close window
        }
    }
    else
    {
        extVisImg = &visImg;
    }
}
