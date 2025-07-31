/*!
 * \copyright Copyright (C) 2025 , LG Electronics, All Right Reserved.
 * \note No part of this source code may be communicated, distributed, reproduced
 * or transmitted in any form or by any means, electronic or mechanical or
 * otherwise, for any purpose, without the prior written permission of
 * LG Electronics.
 * \file    pqcontrollerCustomLutFunction.h
 * \brief   Control of Custom 3D LUT calculation
 * \version 1.0.0
 * \date    2025.06.11
 * \author	hwiyoung.noh
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <unistd.h>
#include <vector>

using namespace std;

#define LUT_STEP_SIZE 17
#define ORIGINAL_LUT_SIZE 33
#define FIXED_POINT_SCALE 100.0f

/*-----------------------------------------------------------------------------
        Type Definitions
------------------------------------------------------------------------------*/
//#define DEBUG
#define ORIGIN

/*-----------------------------------------------------------------------------
        Definitions
------------------------------------------------------------------------------*/
// Define a consistent data structure for polygon coordinates
using GraphCoordinates = std::vector<std::vector<std::vector<std::pair<int, int>>>>;

// Define a constant for the default pair value
const std::pair<int, int> DEFAULT_PAIR = {0.0, 0.0};

std::vector<std::vector<std::vector<std::pair<int, int>>>> original_grpah_coordinate;
std::vector<std::vector<std::vector<std::pair<int, int>>>> prev_graph_coordinate;
std::vector<std::vector<std::vector<std::pair<int, int>>>> current_graph_coordinate;

// Assuming GraphInfo is a defined structure or class
struct GraphInfo {
    int index;
    vector<pair<int, int>> vertices;

    GraphInfo(int idx, vector<pair<int, int>> v) : index(idx), vertices(v) {}
};

typedef struct {
    int gainIndex;
    int row;
    int column;
    int theta;
    int radius;
} COLOR_TUNING_INFO_T;
/*-----------------------------------------------------------------------------
        Global Variable
        (Global Variable Declarations)
------------------------------------------------------------------------------*/

vector<vector<GraphInfo>> graphs; // Global variable for graphs
vector<vector<int>> changed_point_info;
int num_saturations = 8;                 // saturation step
int num_color_angles = 12;               // color angle step
int grid_step    = 2;
int gain_step    = LUT_STEP_SIZE;
int gRadius      = 20;
bool bIsInit     = false;

short cscGamutTable[9] = {0,};
unsigned short gOriginalLutTable[ORIGINAL_LUT_SIZE * ORIGINAL_LUT_SIZE * ORIGINAL_LUT_SIZE * 3] = {0,};
int reDeGammaTable[2] = {0,};
int originalUiValue = 0;
int gGamutTableRgbDomain[ORIGINAL_LUT_SIZE * ORIGINAL_LUT_SIZE * ORIGINAL_LUT_SIZE][3] = {0,};
float gBypassStandardArray[LUT_STEP_SIZE * LUT_STEP_SIZE * LUT_STEP_SIZE][3] = {0,};
bool isLutLoadOk = false;
/*-----------------------------------------------------------------------------
        Code
------------------------------------------------------------------------------*/

// ==================================================== Save / Load =====================================================
bool _load_coordinate_info(void)
{
    // load current_graph_coordinate, prev_graph_coordinate 
    return true;
}

bool _save_coordinate_info(void)
{
    return true;
}

// ==================================================== Main Algorithm ===================================================

float _get_gain(int i) {
    return (float)(i + 1)/(float)LUT_STEP_SIZE;
}

// need to add hsv -> rgb
void hsv_to_rgb(const double hsv[3], double rgb[3]) {
    double h = hsv[0];  // Hue, range [0, 1]
    double s = hsv[1];  // Saturation, range [0, 1]
    double v = hsv[2];  // Value, range [0, 1]

    int Hi = static_cast<int>(std::floor(h * 6.0));  // Hue index
    double f = h * 6.0 - Hi;  // Fractional part
    double p = v * (1.0 - s);
    double q = v * (1.0 - s * f);
    double t = v * (1.0 - (s * (1.0 - f)));

    switch (Hi % 6) {
        case 0:
            rgb[0] = v; rgb[1] = t; rgb[2] = p;
            break;
        case 1:
            rgb[0] = q; rgb[1] = v; rgb[2] = p;
            break;
        case 2:
            rgb[0] = p; rgb[1] = v; rgb[2] = t;
            break;
        case 3:
            rgb[0] = p; rgb[1] = q; rgb[2] = v;
            break;
        case 4:
            rgb[0] = t; rgb[1] = p; rgb[2] = v;
            break;
        case 5:
            rgb[0] = v; rgb[1] = p; rgb[2] = q;
            break;
    }
}

// param : radius, theta, max radius.
std::vector<int> _find_close_lut(double theta, double r, double v, double radius) {
    // Limit the radius to the given maximum
    if (r > radius) {
        r = radius;
    }

    // Convert the angle to hue in HSV color space
    double h = (theta + M_PI) / (2 * M_PI);  // Shift to center at 0
    h = fmod(h + 0.5, 1.0);  // Adjust so red is on the left

    // Normalize the saturation (s) by dividing by the maximum radius
    double s = r / radius;

    // Stack h, s, v as HSV color
    double hsv[3] = {h, s, v};

    // Convert HSV to RGB
    double rgb[3];
    hsv_to_rgb(hsv, rgb);  // You need to implement this function

    std::vector<int> rgbIndex;
    for (int i = 0; i < 3; ++i) {
        // Calculate the base index for each RGB component
        int base = static_cast<int>(rgb[i] / (1.0 / (LUT_STEP_SIZE - 1)));
        double remainder = fmod(rgb[i], 1.0 / (LUT_STEP_SIZE - 1));
        // If the fractional part is more than half the step, round up
        if (remainder / (1.0 / (LUT_STEP_SIZE - 1)) >= 0.5) {
            base += 1;
        }
        rgbIndex.push_back(base);
    }

    return rgbIndex;
}

std::vector<double> _find_lut_hsv(const std::vector<int>& lutVal) {
    std::vector<double> rgb(3);
    for (int i = 0; i < 3; ++i) {
        rgb[i] = static_cast<double>(lutVal[i]) / (LUT_STEP_SIZE - 1);
    }

    double h = 0.0, s = 0.0, v = 0.0;

    // Convert RGB to HSV
    double max = *std::max_element(rgb.begin(), rgb.end());
    double min = *std::min_element(rgb.begin(), rgb.end());
    double delta = max - min;

    if (delta == 0) {
        h = 0;
    } else if (max == rgb[0]) {
        h = fmod(60 * ((rgb[1] - rgb[2]) / delta), 360);
    } else if (max == rgb[1]) {
        h = 60 * ((rgb[2] - rgb[0]) / delta + 2);
    } else if (max == rgb[2]) {
        h = 60 * ((rgb[0] - rgb[1]) / delta + 4);
    }

    if (h < 0) {
        h += 360;
    }

    s = max == 0 ? 0 : delta / max;
    v = max;

    return { h, s, v };
}

std::vector<int> _get_original_lut_idx(int gainIdx, int rowIdx, int columnIdx) {

    double t = (double)original_grpah_coordinate[gainIdx][rowIdx][columnIdx].first / FIXED_POINT_SCALE;
    double r = (double)original_grpah_coordinate[gainIdx][rowIdx][columnIdx].second / FIXED_POINT_SCALE;
    double rgb[3];
    double hsv[3] = {t/M_PI,r/(double)(grid_step * num_saturations),_get_gain(gainIdx)};
    hsv_to_rgb(hsv,rgb);
    printf("[%s] Idx : [%d][%d][%d] value : [%1.3f,%1.3f]. hsv = [%1.2f,%1.2f,%1.2f]. rgb = [%1.3f,%1.3f,%1.3f] \n",__FUNCTION__, gainIdx, rowIdx, columnIdx, t, r,hsv[0],hsv[1],hsv[2],rgb[0],rgb[1],rgb[2]);
    vector<int> lutIndex = _find_close_lut(t,r, _get_gain(gainIdx) ,(double)(grid_step * num_saturations));
    return lutIndex;
}

vector<vector<int>> nearby_points_info; // 인접한 포인트 정보를 저장할 벡터

void _print_lut_differences(void) {
    for (auto a : changed_point_info) {
        int gainIndex = a[0];
        int row = a[1];
        int column = a[2];

        vector<int> originalLutIndex = _get_original_lut_idx(gainIndex, row, column);
        vector<int> newLutIndex = _find_close_lut(
            (double)current_graph_coordinate[gainIndex][row][column].first / FIXED_POINT_SCALE,
            (double)current_graph_coordinate[gainIndex][row][column].second / FIXED_POINT_SCALE,
            _get_gain(gainIndex),
            (double)(grid_step * num_saturations)
        );

        printf("Changed Point LUT: Original = [%d, %d, %d], New = [%d, %d, %d]\n",
            originalLutIndex[0], originalLutIndex[1], originalLutIndex[2],
            newLutIndex[0], newLutIndex[1], newLutIndex[2]);

        // Print differences for nearby points
        for (auto b : nearby_points_info) {
            int nearbyGainIndex = b[0];
            int nearbyRow = b[1];
            int nearbyColumn = b[2];

            vector<int> nearbyOriginalLutIndex = _get_original_lut_idx(nearbyGainIndex, nearbyRow, nearbyColumn);
            vector<int> nearbyNewLutIndex = _find_close_lut(
                (double)current_graph_coordinate[nearbyGainIndex][nearbyRow][nearbyColumn].first / FIXED_POINT_SCALE,
                (double)current_graph_coordinate[nearbyGainIndex][nearbyRow][nearbyColumn].second / FIXED_POINT_SCALE,
                _get_gain(nearbyGainIndex),
                (double)(grid_step * num_saturations)
            );

            printf("Nearby Point LUT: Original = [%d, %d, %d], New = [%d, %d, %d]\n",
                nearbyOriginalLutIndex[0], nearbyOriginalLutIndex[1], nearbyOriginalLutIndex[2],
                nearbyNewLutIndex[0], nearbyNewLutIndex[1], nearbyNewLutIndex[2]);
        }
    }
}

// OK
void _calculate_lut_changed(void) {
    nearby_points_info.clear();

    // All
    for (auto a : changed_point_info) {
        int gainIndex = a[0];
        int row = a[1];
        int column = a[2];
        vector<int> lutIdx = _get_original_lut_idx(a[0],a[1],a[2]);
        printf("Changed Point: GainIndex = %d, Row = %d, Column = %d, LUT = [%d,%d,%d]\n", gainIndex, row, column,lutIdx[0],lutIdx[1],lutIdx[2]);

        // Print nearby points
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                for (int k = -1; k <= 1; k++) {
                    int newGainIndex = gainIndex + i;
                    int newRow = row + j;
                    int newColumn = column + k;

                    // Check bounds
                    if (newGainIndex >= 0 && newGainIndex < gain_step &&
                        newRow >= 0 && newRow < num_color_angles &&
                        newColumn >= 0 && newColumn < num_saturations) {
                        double t = (double)current_graph_coordinate[newGainIndex][newRow][newColumn].first / FIXED_POINT_SCALE;
                        double r = (double)current_graph_coordinate[newGainIndex][newRow][newColumn].second / FIXED_POINT_SCALE;
                        vector<int> lutIndex = _find_close_lut(t, r, _get_gain(newGainIndex), (double)(grid_step * num_saturations));

                        printf("Nearby Point: GainIndex = %d, Row = %d, Column = %d, Theta = %d, Radius = %d, LUT = [%d, %d, %d]\n",
                            newGainIndex, newRow, newColumn,
                            current_graph_coordinate[newGainIndex][newRow][newColumn].first,
                            current_graph_coordinate[newGainIndex][newRow][newColumn].second,
                            lutIndex[0], lutIndex[1], lutIndex[2]);

                        // Store nearby point information
                        nearby_points_info.push_back({newGainIndex, newRow, newColumn});
                    }
                }
            }
        }
    }
}

/*
void _calculate_lut_changed(void)
{
    // All
    for(auto a : changed_point_info) {
        //nearby lut
        //    -> HSV Domain으로 해야 함...
        //    특정 포인트 선택. (Gain, Angle, Saturation)
        int range[3][2] = {{0,}};
        
        if(a[0] >= num_color_angles - 1)    range[0][0]
        
    }

    // or Local
}
*/

void _update_coordinate(void)
{
    printf("_update_coordinate\n");
    int targetIdx = 0;
    for(int i = 0; i < gain_step; i++) {
        if(!(prev_graph_coordinate[i] == current_graph_coordinate[i])) {
            targetIdx = i;
            printf( "change targetIdx : [%2d] \n", i);
            for(int j = 0; j < num_color_angles; j++) {
                for(int k = 0; k < num_saturations; k++) {
                    if(prev_graph_coordinate[i][j][k] != current_graph_coordinate[i][j][k]) {
                        vector<int> orilutIndex =_get_original_lut_idx(i,j,k);
                        double t = (double)current_graph_coordinate[i][j][k].first / FIXED_POINT_SCALE;
                        double r = (double)current_graph_coordinate[i][j][k].second / FIXED_POINT_SCALE;
                        double rgb[3];
                        double hsv[3] = {t/M_PI,r/(double)(grid_step * num_saturations),_get_gain(i)};
                        hsv_to_rgb(hsv,rgb);
                        printf( "Idx : [%d][%d][%d] value : [%1.3f,%1.3f]. hsv = [%1.2f,%1.2f,%1.2f]. rgb = [%1.3f,%1.3f,%1.3f] \n", i, j, k, t, r,hsv[0],hsv[1],hsv[2],rgb[0],rgb[1],rgb[2]);
                        vector<int> lutIndex = _find_close_lut(t,r, _get_gain(i) ,(double)(grid_step * num_saturations));
                        printf( "oriLut = [%d,%d,%d] moved lutIndex = [%d,%d,%d]",orilutIndex[0],orilutIndex[1],orilutIndex[2],lutIndex[0],lutIndex[1],lutIndex[2]);
                        // need to update

                        vector<int> tmpVector = {i,j,k};
                        prev_graph_coordinate[i][j][k] = current_graph_coordinate[i][j][k];
                        changed_point_info.push_back(tmpVector);
                    }
                }
            }
        }
    }
    _calculate_lut_changed();
    changed_point_info.clear();
}

// ==================================================== Initialize =======================================================
void _make_bypass_lut(int option)
{
#ifdef DEBUG
    FILE* outFile = fopen("lutLog_bypass", "w");
    if(!outFile)
        printf( "[%s] file Error \n",__FUNCTION__);
    else
        fprintf(outFile, "Index     R,G,B Value\n");
#endif

    for (int i = 0; i < LUT_STEP_SIZE * LUT_STEP_SIZE * LUT_STEP_SIZE; i++) {
        gBypassStandardArray[i][0] = (float)(i % LUT_STEP_SIZE) / (float)(LUT_STEP_SIZE - 1);
        gBypassStandardArray[i][1] = (float)((i / LUT_STEP_SIZE) % LUT_STEP_SIZE) / (float)(LUT_STEP_SIZE - 1);
        gBypassStandardArray[i][2] = (float)((int)(i / (LUT_STEP_SIZE * LUT_STEP_SIZE)) % LUT_STEP_SIZE) / (float)(LUT_STEP_SIZE - 1);
#ifdef DEBUG
        fprintf(outFile,"[%4d] %1.5f %1.5f %1.5f\n", i, gBypassStandardArray[i][0], gBypassStandardArray[i][1], gBypassStandardArray[i][2]);
#endif
    }

#ifdef DEBUG
        if(outFile)
            fclose(outFile);
#endif
}

void _generate_graphs(void)
{
    // Validate num_color_angles to avoid division by zero
    if (num_color_angles <= 0) {
        printf( "Invalid num_color_angles value: %d\n", num_color_angles);
        return;
    }

    for (int i = 0; i < gain_step; i++) {
        vector<GraphInfo> tmpGraph;
        vector<double> saturations;

        // Precompute saturation radii
        for (int k = 0; k < num_saturations; k++) {
            double radius = grid_step * (k + 1);
            saturations.push_back(radius);
        }

        // Generate angle and create GraphInfo objects
        for (int j = 0; j < num_color_angles; j++) {
            double angle = (2 * M_PI * j) / num_color_angles;

            // Create vertices as (angle, radius) pairs
            vector<pair<int, int>> vertices;
            for (double radius : saturations) {
                vertices.emplace_back(static_cast<int>(angle * FIXED_POINT_SCALE), static_cast<int>(radius * FIXED_POINT_SCALE));
            }

            // Create and add GraphInfo object
            tmpGraph.emplace_back(GraphInfo(i, vertices));
        }

        graphs.emplace_back(tmpGraph);

#ifdef DEBUG
        printf( "============== Gain Step : [%2d] ==================\n", i);
        for (int j = 0; j < num_color_angles; j++) {
            printf( "color_angles = [%d]\n",j);
            for (int k = 0; k < num_saturations; k++) {
                double rgb[3];
                double hsv[3] = {((double)tmpGraph[j].vertices[k].first/100.0f)/(2*M_PI),((double)tmpGraph[j].vertices[k].second/100.0f)/(double)(grid_step * num_saturations),_get_gain(i)};
                hsv_to_rgb(hsv,rgb);
                printf( "saturation_grid : [%2d] coordinate = [%3d,%4d], hsv = [%1.3f,%1.3f,%1.3f] RGB = [%1.3f,%1.3f,%1.3f]\n", k, tmpGraph[j].vertices[k].first,
                           tmpGraph[j].vertices[k].second,hsv[0],hsv[1],hsv[2],rgb[0],rgb[1],rgb[2]);
            }
        }
#endif
    }
}

void pqcontrollerCustomLut_init_custom_lut_function(void)
{
    // Initialize the vectors after _generate_graphs() to avoid redundant initialization
    original_grpah_coordinate = GraphCoordinates(
        gain_step, std::vector<std::vector<std::pair<int, int>>>(
                       num_color_angles, std::vector<std::pair<int, int>>(num_saturations, DEFAULT_PAIR)));
    prev_graph_coordinate = GraphCoordinates(
        gain_step, std::vector<std::vector<std::pair<int, int>>>(
                       num_color_angles, std::vector<std::pair<int, int>>(num_saturations, DEFAULT_PAIR)));
    current_graph_coordinate = GraphCoordinates(
        gain_step, std::vector<std::vector<std::pair<int, int>>>(
                       num_color_angles, std::vector<std::pair<int, int>>(num_saturations, DEFAULT_PAIR)));

    _generate_graphs();

    for (int i = 0; i < gain_step; i++) {
        for (int j = 0; j < num_color_angles; j++) {
            for (int k = 0; k < num_saturations; k++) {
                original_grpah_coordinate[i][j][k] = {graphs[i][j].vertices[k].first,
                                                        graphs[i][j].vertices[k].second};
                prev_graph_coordinate[i][j][k] = {graphs[i][j].vertices[k].first,
                                                    graphs[i][j].vertices[k].second};
                // need to update using load
                current_graph_coordinate[i][j][k] = {graphs[i][j].vertices[k].first,
                                                    graphs[i][j].vertices[k].second};
            }
        }
    }
    // Need to load initial LUT table!
    // /cmn_data/pqcontroller/~
    _make_bypass_lut(0);

    bIsInit = true;
}


 // ==================================================== API =======================================================
/*
    typedef struct {
        int gainIndex;
        int row;
        int column;
        int theta;
        int radius;
    } COLOR_TUNING_INFO_T;
*/

bool pqcontrollerCustomLut_setColorFineParam(vector<COLOR_TUNING_INFO_T> colorInfo, int function)
{
    if(bIsInit != true) {
        printf("Init not finish!\n");
        return false;
    }

#ifdef DEBUG
    printf( "[%s] Enter \n",__FUNCTION__);
#endif

    for( auto a : colorInfo) {
        // Check parameter
        if(a.gainIndex >= (LUT_STEP_SIZE - 1))
            a.gainIndex = LUT_STEP_SIZE - 1;
        if(a.row >= (num_color_angles - 1))
            a.row = num_color_angles - 1;
        if(a.column >= (num_saturations - 1))
            a.column = num_saturations - 1;
        if(a.radius >= (num_saturations * grid_step * FIXED_POINT_SCALE))
            a.radius = num_saturations * grid_step * FIXED_POINT_SCALE;

        graphs[a.gainIndex][a.row].vertices[a.column] = {a.theta, a.radius};
        current_graph_coordinate[a.gainIndex][a.row][a.column] = {a.theta, a.radius};
    }
    
    _update_coordinate();

    return true;
}

bool pqcontrollerCustomLut_getColorFineParam(vector<COLOR_TUNING_INFO_T> &colorInfo)
{
    for(int i = 0; i < gain_step; i++) {
        for(int j = 0; j < num_color_angles; j++) {
            for(int k = 0; k < num_saturations; k++) {
                if(original_grpah_coordinate[i][j][k] != graphs[i][j].vertices[k]) {
                    colorInfo.push_back(COLOR_TUNING_INFO_T{i,j,k,graphs[i][j].vertices[k].first,graphs[i][j].vertices[k].second});
                    printf( "[Diff] graphs[%d][%d].vertices[%d] = [%d,%d] \n", i,j,k,graphs[i][j].vertices[k].first,graphs[i][j].vertices[k].second);
                }
            }
        }
    }
    return true;
}

bool pqcontrollerCustomLut_setColorFineFunction(int function)
{

    if (function == 5) {
        _make_bypass_lut(0);
    }

    return true;
}

void parseJsonString(const std::string& jsonData, COLOR_TUNING_INFO_T& colorInfo) {
    std::istringstream ss(jsonData);
    std::string key, value;

    while (ss >> key) {
        if (key.find("gainIndex") != std::string::npos) {
            ss.ignore(3); // Skip " : "
            ss >> colorInfo.gainIndex;
        } else if (key.find("row") != std::string::npos) {
            ss.ignore(3); // Skip " : "
            ss >> colorInfo.row;
        } else if (key.find("column") != std::string::npos) {
            ss.ignore(3); // Skip " : "
            ss >> colorInfo.column;
        } else if (key.find("theta") != std::string::npos) {
            ss.ignore(3); // Skip " : "
            ss >> colorInfo.theta;
        } else if (key.find("radius") != std::string::npos) {
            ss.ignore(3); // Skip " : "
            ss >> colorInfo.radius;
        }
    }
}

int main(void) {
    pqcontrollerCustomLut_init_custom_lut_function();

    // JSON 데이터 입력
    int size = 4;
    std::string jsonData[size] = {
        R"({"gainIndex" : 10, "row" : 3, "column" : 6, "theta" : 564, "radius" : 1200})",
        R"({"gainIndex" : 2, "row" : 6, "column" : 10, "theta" : 564, "radius" : 1200})",
        R"({"gainIndex" : 1, "row" : 2, "column" : 7, "theta" : 564, "radius" : 1200})",
        R"({"gainIndex" : 4, "row" : 6, "column" : 2, "theta" : 564, "radius" : 1200})",
    };

    for(int i = 0; i < size; i++)
    {
        // COLOR_TUNING_INFO_T 구조체로 변환
        COLOR_TUNING_INFO_T colorInfo;
        parseJsonString(jsonData[i], colorInfo);

        // 벡터에 추가
        std::vector<COLOR_TUNING_INFO_T> colorInfoVec;
        colorInfoVec.push_back(colorInfo);

        // pqcontrollerCustomLut_setColorFineParam 함수 호출
        pqcontrollerCustomLut_setColorFineParam(colorInfoVec, 0);
        usleep( 4000 * 1000 );
    }

    return 0;
}