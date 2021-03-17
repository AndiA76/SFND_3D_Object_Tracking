// ============================================================================
//  
//  Project 2.2: 3D Object Tracking (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
//  
//  Source:      https://github.com/udacity/SFND_3D_Object_Tracking
//
// ============================================================================

// function definitions for exporting 3D object tracking evaluation results to csv file

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>

#include <boost/circular_buffer.hpp>
#include <boost/circular_buffer/base.hpp>

#include "exportResults.hpp"
#include "dataStructures.h"

using namespace std;


// export evaluation results to csv file
void exportResultsToCSV(
    const std::string fullFilename, boost::circular_buffer<EvalResults> & resultBuffer, bool bPrintDebugInfo)
{
    /* export evaluation results to a CSV file with one header line and as many rows as there are images
    *  @param: fullFilename - full filepath to the csv file
    *  @param: resultBuffer - circular buffer holding the results with as many entries as there are images
    *  @param: bPrintDebugInfo - print debug info to screen if true
    */

    // open csv file
    ofstream csv_file;
    csv_file.open(fullFilename, ios::out);

    // write file header using the EvalResults data structure
    stringstream headerStream;
    headerStream << "imgFilename" << ","
                 << "detectorType" << ","
                 << "numKeypoints" << ","
                 << "t_detKeypoints [s]" << ","
                 << "bLimitKpts" << ","
                 << "numKeypointsLimited" << ","
                 << "meanDetectorResponse" << ","
                 << "meanKeypointDiam" << ","
                 << "varianceKeypointDiam_av" << ","
                 << "descExtractorType" << ","
                 << "t_descKeypoints [s]" << ","
                 << "t_detKeypoints + t_descKeypoints [s]" << ","
                 << "matcherType" << ","
                 << "descriptorType" << ","
                 << "selectorType" << ","
                 << "numDescMatches" << ","
                 << "t_matchDescriptors [s]" << ","
                 << "t_detKeypoints + t_descKeypoints + t_matchDescriptors [s]" << ","
                 << "numLidarPointsOnTarget" << ","
                 << "ttcLidar [s]" << ","
                 << "numKptMatchesOnTarget" << ","
                 << "ttcCamera [s]" << ",";
    string headerRow = headerStream.str();
    if (bPrintDebugInfo)
        cout << headerRow << endl; // debug
    // write file header to csv file
    csv_file << headerRow << endl;

    // initialize cumulated sums of keypoints / descriptors / descriptor matches over all images
    int numKeypoints_cumulated = 0;
    int numKeypointsLimited_cumulated = 0;
    int numDescMatches_cumulated = 0;

    // initialize average of mean keypoint detector response over all images
    double meanDetectorResponse_avg = 0.0;

    // initialize average of mean and variance of the keypoint diameter distributions over all images
    double meanKeypointDiam_avg = 0.0;
    double varianceKeypointDiam_avg = 0.0;

    // initialize average processing times over all images
    double t_detKeypoints_avg = 0.0;
    double t_descKeypoints_avg = 0.0;
    double t_sum_det_desc_avg = 0.0;
    double t_matchDescriptors_avg = 0.0;
    double t_sum_det_desc_match_avg = 0.0;

    // initialize average number of 3D Lidar points and 2D keypoints on target object
    int numLidarPointsOnTarget_avg = 0;
    int numKptMatchesOnTarget_avg = 0;

    // counter
    int cnt = 0;

    // loop over the evaluation results for each image / image pair in the result buffer
    for (auto results = resultBuffer.begin(); results != resultBuffer.end(); results++)
    {
        // format next row to write the results per frame to csv file
        stringstream stream;
        stream << results->imgFilename << ","
               << results->detectorType << ","
               << results->numKeypoints << ","
               << fixed << setprecision(6) << results->t_detKeypoints << ","
               << results->bLimitKpts << ","
               << results->numKeypointsLimited << ","
               << fixed << setprecision(6) << results->meanDetectorResponse << ","
               << fixed << setprecision(6) << results->meanKeypointDiam << ","
               << fixed << setprecision(6) << results->varianceKeypointDiam << ","
               << results->descExtractorType << ","
               << fixed << setprecision(6) << results->t_descKeypoints << ","
               << fixed << setprecision(6) << results->t_sum_det_desc << ","
               << results->matcherType << ","
               << results->descriptorType << ","
               << results->selectorType << ","
               << results->numDescMatches << ","
               << fixed << setprecision(6) << results->t_matchDescriptors << ","
               << fixed << setprecision(6) << results->t_sum_det_desc_match << ","
               << results->numLidarPointsOnTarget << ","
               << fixed << setprecision(6) << results->ttcLidar << ","
               << results->numKptMatchesOnTarget << ","
               << fixed << setprecision(6) << results->ttcCamera;
        string next_row = stream.str();
        if (bPrintDebugInfo)
            cout << next_row << endl; // debug
        // write next row to csv file
        csv_file << next_row << endl;

        // cumulate keypoints / descriptors / descriptor matches over all images
        numKeypoints_cumulated += results->numKeypoints;
        numKeypointsLimited_cumulated += results->numKeypointsLimited;
        numDescMatches_cumulated += results->numDescMatches;

        // cumulate mean keypoint detector response over all images
        meanDetectorResponse_avg += results->meanDetectorResponse;

        // cumulate mean and variance of the per image keypoint diameter distribution
        meanKeypointDiam_avg += results->meanKeypointDiam;
        varianceKeypointDiam_avg += results->varianceKeypointDiam;

        // cumulate processing times
        t_detKeypoints_avg += results->t_detKeypoints;
        t_descKeypoints_avg += results->t_descKeypoints;
        t_sum_det_desc_avg += results->t_sum_det_desc;
        t_matchDescriptors_avg += results->t_matchDescriptors;
        t_sum_det_desc_match_avg += results->t_sum_det_desc_match;

        // cumulate number of 3D Lidar points and 2D keypoint matches on target object
        numLidarPointsOnTarget_avg += results->numLidarPointsOnTarget;
        numKptMatchesOnTarget_avg += results->numKptMatchesOnTarget;

        // increment counter
        cnt++;
    }

    // calculate average mean value of detector response over all images
    meanDetectorResponse_avg /= cnt;

    // calculate average keypoint environment (mean and variance) over all images
    meanKeypointDiam_avg /= cnt;
    varianceKeypointDiam_avg /= cnt;

    // calculate average processing times over all iamges
    t_detKeypoints_avg /= cnt;
    t_descKeypoints_avg /= cnt;
    t_sum_det_desc_avg /= cnt;
    t_matchDescriptors_avg /= cnt;
    t_sum_det_desc_match_avg /= cnt;

    // calculate average number of 3D Lidar points and 2D keypoint matches on target object
    numLidarPointsOnTarget_avg /= cnt;
    numKptMatchesOnTarget_avg /= cnt;

    // format bottom row 1 to write the cumulated sums of detected keypoints over all images to csv file
    stringstream stream_1;
    stream_1 << "cumulated sum" << ","
           << "" << ","
           << numKeypoints_cumulated << ","
           << "" << ","
           << "" << ","
           << numKeypointsLimited_cumulated << ","
           << "" << ","
           << "" << ","
           << "" << ","
           << "" << ","
           << "" << ","
           << "" << ","
           << "" << ","
           << "" << ","
           << "" << ","
           << numDescMatches_cumulated << ","
           << "" << ","
           << "" << ","
           << "" << ","
           << "" << ","
           << "" << ","
           << "" << ",";
    string bottom_row_1 = stream_1.str();
    if (bPrintDebugInfo)
        cout << bottom_row_1 << endl; // debug
    // write bottom row 1 to csv file
    csv_file << bottom_row_1 << endl;

    // format bottom row 2 to write the average processing times over all images to csv file
    stringstream stream_2;
    stream_2 << "average values" << ","
           << "" << ","
           << "" << ","
           << fixed << setprecision(6) << t_detKeypoints_avg << ","
           << "" << ","
           << "" << ","
           << fixed << setprecision(6) << meanDetectorResponse_avg << ","
           << fixed << setprecision(6) << meanKeypointDiam_avg << ","
           << fixed << setprecision(6) << varianceKeypointDiam_avg << ","
           << "" << ","
           << fixed << setprecision(6) << t_descKeypoints_avg << ","
           << fixed << setprecision(6) << t_sum_det_desc_avg << ","
           << "" << ","
           << "" << ","
           << "" << ","
           << "" << ","
           << fixed << setprecision(6) << t_matchDescriptors_avg << ","
           << fixed << setprecision(6) << t_sum_det_desc_match_avg << ","
           << numLidarPointsOnTarget_avg << ","
           << "" << ","
           << numKptMatchesOnTarget_avg << ","
           << "" << ",";
    string bottom_row_2 = stream_2.str();
    if (bPrintDebugInfo)
        cout << bottom_row_2 << endl; // debug
    // write bottom row 2 to csv file
    csv_file << bottom_row_2 << endl;

    // close csv file
    csv_file.close();
    
    // print file location where the results have been stored
    cout << "Results have been exported to " << fullFilename << endl;
}


// export overall evaluation results to csv file
void exportOverallResultsToCSV(
    const std::string fullFilename, std::vector<boost::circular_buffer<EvalResults>> & evalResultBuffers, bool bPrintDebugInfo)
{
    /* export overall evaluation results for all image frames and all keypoint detector - descriptor extractor combinations
    *  to a CSV file with one header line, as many rows as there are image frames and as many columns as there are keyptoint 
    *  detector - descriptor extractor combinations
    *  @param: fullFilename - full filepath to the csv file
    *  @param: resultBuffers - vector of circular buffers holding the evaluation results for each detector - descriptor combination
    *  @param: bPrintDebugInfo - print debug info to screen if true
    */

    // open csv file
    ofstream csv_file;
    csv_file.open(fullFilename, ios::out);

    // write file header using the EvalResults data structure
    stringstream headerStream;
    headerStream << "imgFilename" << ","
                 << "numLidarPointsOnTarget" << ","
                 << "ttcLidar [s]" << ",";
    // loop over the vector of result buffers to get all detector / descriptor combinations
    int id = 1; // initialize detector - descriptor combination id
    for (auto resBuf = evalResultBuffers.begin(); resBuf != evalResultBuffers.end(); resBuf++)
    {
        // add two columns for each keypoint detector - descriptor extractor combination
        headerStream << "numKptMatchesOnTarget (" << id << ": " << ((*resBuf).begin())->detectorType << "/" << ((*resBuf).begin())->descExtractorType << "),"
                     << "ttcCamera [s] (" << id << ": " << ((*resBuf).begin())->detectorType << "/" << ((*resBuf).begin())->descExtractorType << "),";

        // increment detector - descriptor combination id
        id++;
    }
    string headerRow = headerStream.str();
    if (bPrintDebugInfo)
        cout << headerRow << endl; // debug
    // write file header to csv file
    csv_file << headerRow << endl;

    // get number of image frames
    int numOfImgFrames = (*evalResultBuffers.begin()).size();
    cout << "Number of image frames stored in result buffer: " << numOfImgFrames << endl;

    // loop over the vector of result buffers for each detector - descriptor combination
    for (auto i = 0; i != numOfImgFrames; i++)
    {
        // reset detector - descriptor combination id
        id = 1;
        // write next row using the EvalResults data structure
        stringstream stream;
        stream << ((*evalResultBuffers.begin())[i]).imgFilename << ","
               << ((*evalResultBuffers.begin())[i]).numLidarPointsOnTarget << ","
               << fixed << setprecision(6) << ((*evalResultBuffers.begin())[i]).ttcLidar << ",";
        // loop over all detector - descriptor combinations
        for (auto resBuf = evalResultBuffers.begin(); resBuf != evalResultBuffers.end(); resBuf++)
        {
            // add two columns for each keypoint detector - descriptor extractor combination
            stream << ((*resBuf)[i]).numKptMatchesOnTarget << ","
                   << fixed << setprecision(6) << ((*resBuf)[i]).ttcCamera << ",";
            
            // increment detector / descriptor combination id
            ++id;
        }
        string next_row = stream.str();
        if (bPrintDebugInfo)
            cout << next_row << endl; // debug
        // write next row to csv file
        csv_file << next_row << endl;
    }

    // close csv file
    csv_file.close();
    
    // print file location where the results have been stored
    cout << "Overall results have been exported to " << fullFilename << endl;
}
