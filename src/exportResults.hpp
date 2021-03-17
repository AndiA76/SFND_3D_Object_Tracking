// ============================================================================
//  
//  Project 2.2: 3D Object Tracking (Udacity Sensor Fusion Nanodegree)
// 
//  Authors:     Andreas Albrecht using code base/skeleton provided by Udacity
//  
//  Source:      https://github.com/udacity/SFND_3D_Object_Tracking
//
// ============================================================================

// function declarations for exporting 3D object tracking evaluation results to csv file

#ifndef exportResults_hpp
#define exportResults_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>

#include <boost/circular_buffer.hpp>
#include <boost/circular_buffer/base.hpp>

#include "dataStructures.h"

void exportResultsToCSV(
    const std::string fullFilename,
    boost::circular_buffer<EvalResults> & resultBuffer,
    bool bPrintDebugInfo = false);
    
void exportOverallResultsToCSV(
    const std::string fullFilename,
    std::vector<boost::circular_buffer<EvalResults>> & evalResultBuffers,
    bool bPrintDebugInfo = false);

#endif /* exportResults_hpp */