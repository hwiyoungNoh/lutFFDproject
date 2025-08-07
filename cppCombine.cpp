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

/*
#include "logging.h"
#include "pqcontrollerCustomLutFunction.h"
#include "pqcontroller.h"
#include "xplatform.h"
#include "pqcontrollerMenuData.h"
*/

using namespace std;

#define LUT_STEP_SIZE 17
#define ORIGINAL_LUT_SIZE 33
#define FIXED_POINT_SCALE 100.0f
#define DEBUG
#define ALGORITHM_VERSION_1
//#define SUPPORT_GAIN_INTERPOLATION

typedef struct {
    int gainIndex;
    int row;
    int column;
    int theta;
    int radius;
} COLOR_TUNING_INFO_T;

// ----------------- 자료구조/글로벌 -----------------
struct GraphInfo {
    int index;
    vector<pair<int, int>> vertices; // (theta, radius/fixed-point)
    GraphInfo(int idx, vector<pair<int, int>> v) : index(idx), vertices(v) {}
};
 
using GraphCoordinates = vector<vector<vector<pair<int, int>>>>;
const pair<int, int> DEFAULT_PAIR = {0,0};
 
vector<vector<GraphInfo>> graphs;
GraphCoordinates original_grpah_coordinate, prev_graph_coordinate, current_graph_coordinate;
vector<vector<int>> changed_point_info;
 
int num_color_angles = 12;
int num_saturations = 8;
int grid_step = 2;
int saturation_max_level = grid_step * num_saturations;

int gain_step = LUT_STEP_SIZE;
int gRadius = 20;
bool bIsInit = false;

short cscGamutTable[9] = {0,};
//unsigned short gOriginalLutTable[ORIGINAL_LUT_SIZE * ORIGINAL_LUT_SIZE * ORIGINAL_LUT_SIZE * 3] = {0,};
int reDeGammaTable[2] = {0,};
unsigned int gOriginalLutTable[LUT_STEP_SIZE * LUT_STEP_SIZE * LUT_STEP_SIZE * 3] = {0,};
int originalUiValue = 0;
int gGamutTableRgbDomain[ORIGINAL_LUT_SIZE * ORIGINAL_LUT_SIZE * ORIGINAL_LUT_SIZE][3] = {0,};
float gBypassStandardArray[LUT_STEP_SIZE * LUT_STEP_SIZE * LUT_STEP_SIZE][3] = {0,};
float gCurrentGamutArray[LUT_STEP_SIZE * LUT_STEP_SIZE * LUT_STEP_SIZE][3] = {0,};
unsigned short gSoc17LutTable[LUT_STEP_SIZE * LUT_STEP_SIZE * LUT_STEP_SIZE * 3] = {0,};
unsigned short gSocOutputTable[ORIGINAL_LUT_SIZE * ORIGINAL_LUT_SIZE * ORIGINAL_LUT_SIZE * 3] = {0,};
float final_data[ORIGINAL_LUT_SIZE*ORIGINAL_LUT_SIZE*ORIGINAL_LUT_SIZE * 3] = {0.0f,};
float final17_data[LUT_STEP_SIZE*LUT_STEP_SIZE*LUT_STEP_SIZE * 3] = {0.0f,};
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

bool _load_original_3dlut(int select)
{
    // Get original 3D LUT
    // Need to consider 3 x 3 matrix
    int lutSize = 0;
    //originalUiValue = controller_fineColor_get3dLutData(cscGamutTable,gOriginalLutTable,&lutSize,reDeGammaTable);
	
    printf( "[%s] originalUiValue %d, lutSize = %d \n",__FUNCTION__,originalUiValue,lutSize);
	
#ifdef DEBUG
    FILE* outFile = fopen("/var/log/lutLog", "w");
    
    if(!outFile)
        printf( "[%s] file Error \n",__FUNCTION__);
    else
        fprintf(outFile, "Index     R,G,B Value\n");

#endif
    for(int i = 0; i < lutSize * lutSize * lutSize; i++)
    {
        gGamutTableRgbDomain[i][0] = (int)gOriginalLutTable[(3 * i) + 0];
        gGamutTableRgbDomain[i][1] = (int)gOriginalLutTable[(3 * i) + 1];
        gGamutTableRgbDomain[i][2] = (int)gOriginalLutTable[(3 * i) + 2];
#ifdef DEBUG
        if(outFile)
            fprintf(outFile, "[%5d] %5d, %5d, %5d,  [%5d,%5d,%5d] \n", i,gGamutTableRgbDomain[i][0],gGamutTableRgbDomain[i][1],gGamutTableRgbDomain[i][2],gOriginalLutTable[(3 * i) + 0],gOriginalLutTable[(3 * i) + 1],gOriginalLutTable[(3 * i) + 2]);
#endif
    }

#ifdef DEBUG
        if(outFile)
            fclose(outFile);
#endif
    isLutLoadOk = true;
    
	return true;
}

void _make_soc_3dlut(void) {
    for (int i = 0; i < (LUT_STEP_SIZE * LUT_STEP_SIZE * LUT_STEP_SIZE); i++) {
        gSoc17LutTable[i*3 + 0] = (unsigned short)(gCurrentGamutArray[i][0] * 4095);
        gSoc17LutTable[i*3 + 1] = (unsigned short)(gCurrentGamutArray[i][1] * 4095);
        gSoc17LutTable[i*3 + 2] = (unsigned short)(gCurrentGamutArray[i][2] * 4095);
    }
}

//============= HSV <-> RGB 변환 =============//
void rgb_to_hsv(double r, double g, double b, double& h, double& s, double& v) {
    double M = max(r, max(g, b));
    double m = min(r, min(g, b));
    v = M;
    double c = M - m;
    s = (v > 0) ? c / v : 0;
    h = 0;
    if (c != 0) {
        if (M == r)      h = fmod((g - b)/c,6.0)/6.0;
        else if (M == g) h = ((b - r)/c + 2.0)/6.0;
        else             h = ((r - g)/c + 4.0)/6.0;
    }
    if (h < 0) h += 1.0;
}
void hsv_to_rgb(const double hsv[3], double rgb[3]) {
    double h=hsv[0], s=hsv[1], v=hsv[2];
    int Hi = int(floor(h*6.0));
    double f = h*6.0 - Hi, p = v*(1-s), q = v*(1-s*f), t = v*(1-(s*(1-f)));
    switch (Hi%6) {
        case 0: rgb[0]=v; rgb[1]=t; rgb[2]=p; break;
        case 1: rgb[0]=q; rgb[1]=v; rgb[2]=p; break;
        case 2: rgb[0]=p; rgb[1]=v; rgb[2]=t; break;
        case 3: rgb[0]=p; rgb[1]=q; rgb[2]=v; break;
        case 4: rgb[0]=t; rgb[1]=p; rgb[2]=v; break;
        case 5: rgb[0]=v; rgb[1]=p; rgb[2]=q; break;
    }
}
inline double get_gain(int idx) { return (double)(idx)/(double)(LUT_STEP_SIZE-1); }

int getLinearArrayIndex(int r, int g, int b, int size) {
    return (b * size * size) + (g * size) + r;
}

pair<double,double> polar_to_cart(double h, double s) {
    double theta = h*2*M_PI, r=s; return make_pair(r*cos(theta), r*sin(theta));
}
bool point_in_polygon(double x, double y, const vector<pair<double,double>>& poly) {
    int n=poly.size(); bool in=false;
    for(int i=0,j=n-1;i<n;j=i++) {
        double xi=poly[i].first,yi=poly[i].second, xj=poly[j].first,yj=poly[j].second;
        if((yi>y)!=(yj>y)&&(x<(xj-xi)*(y-yi)/((yj-yi)+1e-12)+xi)) in=!in;
    }
    return in;
}
std::vector<int> find_close_lut(double theta, double r, double v, double radius, int size) {
    if (r > radius) {
        r = radius;
    }

    double h = theta / (2 * M_PI);
    h = std::fmod(h, 1.0);

    double hsv[3] = { h, r / radius, v };
    double rgb[3];
    hsv_to_rgb(hsv, rgb);

    std::vector<int> rgbIndex(3);
    for (int i = 0; i < 3; ++i) {
        int base = static_cast<int>(rgb[i] / (1.0 / (size - 1)));
        if ((fmod(rgb[i], (1.0 / (size - 1)))) / (1.0 / (size - 1)) >= 0.5) {
            base += 1;
        }
        rgbIndex[i] = base;
    }
#ifdef DEBUG
    printf("[%s] input = [%.4f, %2.4f,%.4f], hsv=( %.4f,%.4f,%.4f ) rgb = (%.4f,%.4f,%.4f)\n",
        __FUNCTION__,theta,r,v,hsv[0],hsv[1],hsv[2],rgb[0],rgb[1],rgb[2]);
#endif

    return rgbIndex;
}

#ifdef ALGORITHM_VERSION_1
struct Ctl { int gain, axis, level; int lutIndex[3];};
vector<Ctl> get_adjacent_controls(int gain, int axis, int level) {
    vector<Ctl> adj;
#ifdef SUPPORT_GAIN_INTERPOLATION
    for (int dg = -1; dg <= 1; ++dg) {
        int ng = gain + dg;
        if (ng < 0 || ng >= LUT_STEP_SIZE) continue; // 순환 X, 경계
        for (int da = -1; da <= 1; ++da) {
            // axis(row): 0~11에서 순환
            int na = (axis + da + num_color_angles) % num_color_angles;
            for (int dl = -1; dl <= 1; ++dl) {
                int nl = level + dl;
                if (nl < 0 || nl >= num_saturations) continue; // 순환 X
                if (dg == 0 && da == 0 && dl == 0) continue;
                vector<int> tmp = find_close_lut(theta,r,v,saturation_max_level,LUT_STEP_SIZE);
                adj.push_back({ng, na, nl, {tmp[0],tmp[1],tmp[2]}});
            }
        }
    }
#else
#ifdef DEBUG
    printf("[%s] Find adjacent points. original = [%d,%d,%d]\n",__FUNCTION__,gain,axis,level);
#endif
    //control only same gain level
    for (int da = -1; da <= 1; ++da) {
        // axis(row): 0~11에서 순환
        int na = (axis + da + num_color_angles) % num_color_angles;
        for (int dl = -1; dl <= 1; ++dl) {
            int nl = level + dl;
            if (nl < 0 || nl >= num_saturations) continue; // 순환 X
            if (da == 0 && dl == 0) continue;
            vector<int> tmp = find_close_lut((float)original_grpah_coordinate[gain][na][nl].first / FIXED_POINT_SCALE,(float)original_grpah_coordinate[gain][na][nl].second / FIXED_POINT_SCALE,get_gain(gain),(float)saturation_max_level,LUT_STEP_SIZE);
            adj.push_back({gain, na, nl, {tmp[0],tmp[1],tmp[2]}});
        }
    }
#endif
    return adj;
}
#else
struct Ctl { int gain, axis, level;};
vector<Ctl> get_adjacent_controls(int gain, int axis, int level) {
    vector<Ctl> adj;
    for (int dg = -1; dg <= 1; ++dg) {
        int ng = gain + dg;
        if (ng < 0 || ng >= LUT_STEP_SIZE) continue; // 순환 X, 경계
        for (int da = -1; da <= 1; ++da) {
            // axis(row): 0~11에서 순환
            int na = (axis + da + num_color_angles) % num_color_angles;
            for (int dl = -1; dl <= 1; ++dl) {
                int nl = level + dl;
                if (nl < 0 || nl >= num_saturations) continue; // 순환 X
                if (dg == 0 && da == 0 && dl == 0) continue;
                adj.push_back({ng, na, nl, });
            }
        }
    }
    return adj;
}
#endif

#ifdef ALGORITHM_VERSION_1

#else
//============= 보간: FFD 방식 =============//
void smooth_deform_point_3d_interpolated(
    double x, double y, double z,
    double orig_h, double orig_s, double orig_v,
    double new_h, double new_s, double new_v,
    double& out_h, double& out_s, double& out_v
) {
    double d = sqrt((x-orig_h)*(x-orig_h) + (y-orig_s)*(y-orig_s) + (z-orig_v)*(z-orig_v));
    double sharpness = 8.0;
    double alpha = exp(-d*sharpness);
    out_h = x*(1-alpha) + new_h*alpha;
    out_s = y*(1-alpha) + new_s*alpha;
    out_v = z*(1-alpha) + new_v*alpha;
}
#endif

//============= sleep 함수 =============//
//void sleep_millisec(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
 
//============= JSON 파싱 =============//
void parseJsonString(const string& jsonData, COLOR_TUNING_INFO_T& colorInfo) {
    istringstream ss(jsonData); string key;
    while (ss >> key) {
        if (key.find("gainIndex") != string::npos) { ss.ignore(3); ss >> colorInfo.gainIndex; }
        else if (key.find("row") != string::npos) { ss.ignore(3); ss >> colorInfo.row; }
        else if (key.find("column") != string::npos) { ss.ignore(3); ss >> colorInfo.column; }
        else if (key.find("theta") != string::npos) { ss.ignore(3); ss >> colorInfo.theta; }
        else if (key.find("radius") != string::npos) { ss.ignore(3); ss >> colorInfo.radius; }
    }
}

//============= 1. 제어점 이동 및 DEBUG =============//
void apply_control_point_updates(const vector<COLOR_TUNING_INFO_T>& colorInfos) {
    for(auto &a : colorInfos) {
        double orig_theta = original_grpah_coordinate[a.gainIndex][a.row][a.column].first / FIXED_POINT_SCALE;
        double orig_radius = original_grpah_coordinate[a.gainIndex][a.row][a.column].second / FIXED_POINT_SCALE;
        double orig_h = orig_theta / (2 * M_PI);
        double orig_s = orig_radius / saturation_max_level;
        double new_theta = a.theta / double(FIXED_POINT_SCALE);
        double new_radius = a.radius / double(FIXED_POINT_SCALE);
        double new_h = new_theta / (2 * M_PI);
        double new_s = new_radius / saturation_max_level;
        double v = get_gain(a.gainIndex);
 
        graphs[a.gainIndex][a.row].vertices[a.column] = {a.theta, a.radius};
        current_graph_coordinate[a.gainIndex][a.row][a.column] = {a.theta, a.radius};
#ifdef DEBUG
        printf("[CONTROL] idx=(%2d,%2d,%2d) orig_hsv=(%.4f,%.4f,%.4f) --> moved_hsv=(%.4f,%.4f,%.4f)\n",
            a.gainIndex, a.row, a.column, orig_h, orig_s, v, new_h, new_s, v);
#endif
    }
}
 
//============= 2. 변경 제어점만 추출 및 DEBUG =============//
vector<vector<int>> extract_changed_control_points() {
    vector<vector<int>> changed;
    for(int i=0;i<gain_step;i++)
    for(int j=0;j<num_color_angles;j++)
    for(int k=0;k<num_saturations;k++){
        if(current_graph_coordinate[i][j][k]!=prev_graph_coordinate[i][j][k]) {
            changed.push_back({i,j,k});
#ifdef DEBUG
            double orig_theta = original_grpah_coordinate[i][j][k].first / FIXED_POINT_SCALE;
            double orig_radius = original_grpah_coordinate[i][j][k].second / FIXED_POINT_SCALE;
            double orig_h = orig_theta / (2 * M_PI);
            double orig_s = orig_radius / saturation_max_level;
            double new_theta = current_graph_coordinate[i][j][k].first / double(FIXED_POINT_SCALE);
            double new_radius = current_graph_coordinate[i][j][k].second / double(FIXED_POINT_SCALE);
            double new_h = new_theta / (2 * M_PI);
            double new_s = new_radius / saturation_max_level;
            printf("[CHANGE] idx=[%2d,%2d,%2d] orig_hsv:(%.4f,%.4f,%.4f) -> moved_hsv:(%.4f,%.4f,%.4f)\n",
                i,j,k,orig_h,orig_s,(float)i/(float)(gain_step-1),new_h,new_s,(float)i/(float)(gain_step-1));
#endif
        }
    }
    return changed;
}

// ========================================================================================= Algorithm ==========================================================================

#ifdef ALGORITHM_VERSION_1
void domain_change(const std::vector<int>& near, const std::vector<int>& ori, const std::vector<int>& target, float* lutArray, unsigned int *originalLut, int lutSize) {
/*
    original point - near point -> R, G, B 3 point
    moved point - near point -> R, G, B 3 point
*/
    std::vector<int> oDiff(3), tDiff(3);
    for (int i = 0; i < 3; ++i) {
        oDiff[i] = near[i] - ori[i];
        tDiff[i] = near[i] - target[i];
#ifdef DEBUG
        printf("[%d] near : [%2d], ori : [%2d], tar : [%2d], oDiff : [%2d], tDiff : [%2d]\n",i,near[i],ori[i],target[i],oDiff[i],tDiff[i]);
#endif
    }
    std::vector<std::vector<int>> oTmpList(3);
    for (int i = 0; i < 3; ++i) {
        if (oDiff[i] == 0) {
            oTmpList[i].push_back(near[i]);
        } else if (oDiff[i] < 0) {
            for (int j = 0; j <= -oDiff[i]; ++j) {
                oTmpList[i].push_back(near[i] + j);
            }
        } else {
            for (int j = 0; j <= oDiff[i]; ++j) {
                oTmpList[i].push_back(near[i] - j);
            }
        }
    }

    std::vector<std::vector<int>> oLutList;
    for (int i : oTmpList[0]) {
        for (int j : oTmpList[1]) {
            for (int k : oTmpList[2]) {
                oLutList.push_back({ i, j, k });
            }
        }
    }

    if (oLutList.size() >= 3) {
        oLutList.pop_back();
        oLutList.erase(oLutList.begin());
    }

    std::vector<std::vector<double>> oLutRatio;
    for (const auto& oLut : oLutList) {
        std::vector<double> tmpRatio(3);
        for (int i = 0; i < 3; ++i) {
            if (oDiff[i] == 0) {
                tmpRatio[i] = 1.0;
            } else {
                tmpRatio[i] = static_cast<double>(std::abs(near[i] - oLut[i])) / std::abs(oDiff[i]);
            }
        }
        oLutRatio.push_back(tmpRatio);
    }

    std::vector<std::vector<int>> tTmpList(3);
    for (int i = 0; i < 3; ++i) {
        if (tDiff[i] == 0) {
            tTmpList[i].push_back(near[i]);
        } else if (tDiff[i] < 0) {
            for (int j = 0; j <= -tDiff[i]; ++j) {
                tTmpList[i].push_back(near[i] + j);
            }
        } else {
            for (int j = 0; j <= tDiff[i]; ++j) {
                tTmpList[i].push_back(near[i] - j);
            }
        }
    }

    std::vector<double> tRatio(3);
    for (int i = 0; i < 3; ++i) {
        if (!tTmpList[i].empty()) {
            if (tDiff[i] != 0) {
                tRatio[i] = 1.0 / std::abs(tDiff[i]);
            } else {
                tRatio[i] = 0.0;
            }
        } else {
            tRatio[i] = 0.0;
        }
    }

    for (size_t idx = 0; idx < oLutList.size(); ++idx) {
        std::vector<int> tLut(3);
        for (int rgb = 0; rgb < 3; ++rgb) {
            if (std::abs(tRatio[rgb]) <= 1e-5) {
                tLut[rgb] = near[rgb];
            } else if (std::abs(tRatio[rgb] - 1.0) <= 1e-5) {
                if (oLutRatio[idx][rgb] >= 0.5) {
                    tLut[rgb] = target[rgb];
                } else {
                    tLut[rgb] = near[rgb];
                }
            } else {
                if (std::abs(oLutRatio[idx][rgb] - 1.0) <= 1e-5) {
                    tLut[rgb] = target[rgb];
                } else if (std::abs(oLutRatio[idx][rgb]) <= 1e-5) {
                    tLut[rgb] = near[rgb];
                } else {
                    double resultVal = 0.0;
                    if (near[rgb] - target[rgb] > 0) {
                        resultVal = std::round(near[rgb] - (oLutRatio[idx][rgb] * std::abs(tDiff[rgb])));
                    } else {
                        resultVal = std::round(near[rgb] + (oLutRatio[idx][rgb] * std::abs(tDiff[rgb])));
                    }
                    tLut[rgb] = static_cast<int>(resultVal);
                }
            }
        }

        int arrayTarget = getLinearArrayIndex(tLut[0], tLut[1], tLut[2], lutSize);
        int originalIdx = getLinearArrayIndex(oLutList[idx][0], oLutList[idx][1], oLutList[idx][2], lutSize);
        final_data[originalIdx] = (float)originalLut[arrayTarget]/4095.0f;
    }
}

/*
void update_lut_gain(int polygon_index, int vertex_index, double target_r, double target_theta, std::vector<PolygonInfo>& polygons, std::vector<std::vector<std::vector<double>>>& lutArray, const std::vector<std::vector<std::vector<double>>>& originalLut, int LUTSIZE, double volume, int num_polygons, int num_vertices) {
    double ori_r = polygons[polygon_index].vertices[vertex_index].first;
    double ori_theta = polygons[polygon_index].vertices[vertex_index].second;
    std::vector<int> originalLutIndex = find_close_lut(ori_theta, ori_r, volume, num_polygons * 2, LUTSIZE);

    std::vector<int> targetLutIndex = find_close_lut(target_theta, target_r, volume, num_polygons * 2, LUTSIZE);

    std::vector<std::vector<int>> originalLUT(3, std::vector<int>(3));
    std::vector<std::vector<int>> targetLUT(3, std::vector<int>(3));
    for (int i = 0; i < 3; ++i) {
        int pIdx = i - 1 + polygon_index;
        if (pIdx < 0) {
            pIdx = 0;
        } else if (pIdx >= num_polygons) {
            pIdx = num_polygons - 1;
        }
        for (int j = 0; j < 3; ++j) {
            int vIdx = ((j - 1 + vertex_index) + num_vertices) % num_vertices;
            originalLUT[i][j] = find_close_lut(polygons[pIdx].vertices[vIdx].second, polygons[pIdx].vertices[vIdx].first, volume, num_polygons * 2, LUTSIZE)[0];
            targetLUT[i][j] = find_close_lut(polygons[pIdx].vertices[vIdx].second, polygons[pIdx].vertices[vIdx].first, volume, num_polygons * 2, LUTSIZE)[0];
        }
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i == 1 && j == 1) {
                continue;
            }
            domain_change({ originalLUT[i][j], originalLUT[1][1], targetLUT[1][1] }, { originalLUT[1][1], originalLUT[1][1], targetLUT[1][1] }, { targetLUT[1][1], targetLUT[1][1], targetLUT[1][1] }, lutArray, originalLut, LUTSIZE);
        }
    }
    lutArray[getLinearArrayIndex(originalLUT[0][0], originalLUT[1][1], originalLUT[2][2], LUTSIZE)] = originalLut[getLinearArrayIndex(targetLUT[0][0], targetLUT[1][1], targetLUT[2][2], LUTSIZE)];
    saveLUTfile(lutArray);
}
*/
#endif

//============= 3. 보간 및 gridData 관련 DEBUG =============//
void sector_polygon_and_lut_update(int gain, int row, int col) {
    double orig_theta = original_grpah_coordinate[gain][row][col].first / FIXED_POINT_SCALE;
    double orig_radius = original_grpah_coordinate[gain][row][col].second / FIXED_POINT_SCALE;
    double orig_h = orig_theta / (2 * M_PI), orig_s = orig_radius / saturation_max_level, orig_v = get_gain(gain);
 
    double moved_theta = current_graph_coordinate[gain][row][col].first / FIXED_POINT_SCALE;
    double moved_radius = current_graph_coordinate[gain][row][col].second / FIXED_POINT_SCALE;
    double moved_h = moved_theta / (2 * M_PI), moved_s = moved_radius / saturation_max_level, moved_v = get_gain(gain);
 
    vector<int> oriLutIdx = find_close_lut(orig_h,orig_radius,orig_v,(float)saturation_max_level,LUT_STEP_SIZE);
    vector<int> targetLutIdx = find_close_lut(moved_h,moved_radius,moved_v,(float)saturation_max_level,LUT_STEP_SIZE);
#ifdef DEBUG
    printf("[SECTOR] idx=[%d,%d,%d] orig_hsv=(%.4f,%.4f,%.4f), moved_hsv=(%.4f,%.4f,%.4f), ori_lut_idx = [%d,%d,%d] moved_lut_idx=[%d,%d,%d]\n",
        gain, row, col, orig_h, orig_s, orig_v, moved_h, moved_s, moved_v,oriLutIdx[0],oriLutIdx[1],oriLutIdx[2],targetLutIdx[0],targetLutIdx[1],targetLutIdx[2]);
/*
    printf("[STEP Guide] STEP SIZE  = %d\n",LUT_STEP_SIZE);
    for(int i = 0; i < LUT_STEP_SIZE; i++)
        printf("[%2d] %.5f\n",i,(float)i/(float)(LUT_STEP_SIZE-1));
*/
#endif
 
#ifdef ALGORITHM_VERSION_1
    // -> ? Need to fix!!!!
    vector<Ctl> adj_ctls = get_adjacent_controls(gain, row, col);

#ifdef DEBUG
    printf("[get_adjacent_controls]\n");
    for(int i = 0; i<adj_ctls.size(); i++)
        printf("[%2d] gain : %2d, axis : %2d, level : %2d. RGB LUT Index = [%d,%d,%d] \n",i,adj_ctls[i].gain,adj_ctls[i].axis,adj_ctls[i].level,adj_ctls[i].lutIndex[0],adj_ctls[i].lutIndex[1],adj_ctls[i].lutIndex[2]);
#endif
    //SET Original, target, adjacent points.
    for (int i = 0; i <adj_ctls.size(); ++i)
        domain_change({adj_ctls[i].lutIndex[0],adj_ctls[i].lutIndex[1],adj_ctls[i].lutIndex[2]},oriLutIdx,targetLutIdx,final17_data,gOriginalLutTable,LUT_STEP_SIZE);

#else
    vector<Ctl> adj_ctls = get_adjacent_controls(gain, row, col);
#ifdef DEBUG
    printf("[get_adjacent_controls]\n");
    for(int i = 0; i<adj_ctls.size(); i++) {
        printf("[%2d] gain : %2d, axis : %2d, level : %2d, \n",i,adj_ctls[i].gain,adj_ctls[i].axis,adj_ctls[i].level);
    }
#endif

    vector<pair<double,double>> poly_cart;
    poly_cart.push_back(polar_to_cart(moved_h, moved_s));
    for(auto ctl : adj_ctls) {
        double t = current_graph_coordinate[ctl.gain][ctl.axis][ctl.level].first / FIXED_POINT_SCALE;
        double r = current_graph_coordinate[ctl.gain][ctl.axis][ctl.level].second / FIXED_POINT_SCALE;
        double h = t/(2*M_PI), s = r/saturation_max_level;
        poly_cart.push_back(polar_to_cart(h,s));
    }
    vector<int> all_levels; all_levels.push_back(col); for(auto c: adj_ctls) all_levels.push_back(c.level);
    int min_lv = *min_element(all_levels.begin(),all_levels.end()), max_lv = *max_element(all_levels.begin(),all_levels.end());
 
    int cnt=0;
    for(int r=0;r<LUT_STEP_SIZE;r++) {
        for(int g=0;g<LUT_STEP_SIZE;g++) {
            for(int b=0;b<LUT_STEP_SIZE;b++) {
                double gh,gs,gv;
                rgb_to_hsv(r/16.0, g/16.0, b/16.0, gh, gs, gv);
                auto cart = polar_to_cart(gh,gs);
                int grid_lv = std::round(gs*(num_saturations-1));
                if(grid_lv<min_lv || grid_lv>max_lv) continue;
                if(!point_in_polygon(cart.first, cart.second, poly_cart)) continue;
                double gh2,gs2,gv2;
                smooth_deform_point_3d_interpolated(gh,gs,gv, orig_h, orig_s, orig_v, moved_h, moved_s, moved_v, gh2,gs2,gv2);
                double rgb_new[3], hsv_new[3] = {gh2, gs2, gv2};
                hsv_to_rgb(hsv_new, rgb_new);
                int rr = std::round(rgb_new[0]*16), gg = std::round(rgb_new[1]*16), bb = std::round(rgb_new[2]*16);
                rr = min(max(rr,0),16); gg = min(max(gg,0),16); bb = min(max(bb,0),16);

            #ifdef DEBUG
                if(cnt++<10)
                    printf(" [GRID] RGB idx=(%2d,%2d,%2d) hsv(%.3f,%.3f,%.3f)->(%.3f,%.3f,%.3f) rgb(%.2f,%.2f,%.2f)->LUT(%d,%d,%d)\n",
                        r,g,b,gh,gs,gv,gh2,gs2,gv2, rgb_new[0],rgb_new[1],rgb_new[2], rr,gg,bb);
                else if(cnt==11) printf("...\n");
                printf("Total cnt : %d\n",cnt);
                cnt = 0;
            #endif

                // find changed lut
                if((r != rr) || ( g != gg) || (b != bb)) {
                    gCurrentGamutArray[r + LUT_STEP_SIZE*(g + b*LUT_STEP_SIZE)][0] = gCurrentGamutArray[rr + LUT_STEP_SIZE*(gg + bb*LUT_STEP_SIZE)][0];
                    gCurrentGamutArray[r + LUT_STEP_SIZE*(g + b*LUT_STEP_SIZE)][1] = gCurrentGamutArray[rr + LUT_STEP_SIZE*(gg + bb*LUT_STEP_SIZE)][1];
                    gCurrentGamutArray[r + LUT_STEP_SIZE*(g + b*LUT_STEP_SIZE)][2] = gCurrentGamutArray[rr + LUT_STEP_SIZE*(gg + bb*LUT_STEP_SIZE)][2];  
                    #ifdef DEBUG
                        printf("Diff save [%d,%d,%d] -> [%d,%d,%d]\n",r,g,b,rr,gg,bb);
                    #endif
                }
            }
        }
    }
#endif

}
void interpolateLUT(float* new_values, int inputSize, int outputSize) {
    auto interpolate = [](float start, float end, int steps) {
        std::vector<float> result(steps);
        float step = (end - start) / (steps - 1);
        for (int i = 0; i < steps; ++i) {
            result[i] = start + i * step;
        }
        return result;
    };

    auto x = interpolate(0.0f, 1.0f, inputSize);
    auto y = interpolate(0.0f, 1.0f, inputSize);
    auto z = interpolate(0.0f, 1.0f, inputSize);

    auto new_x = interpolate(0.0f, 1.0f, outputSize);
    auto new_y = interpolate(0.0f, 1.0f, outputSize);
    auto new_z = interpolate(0.0f, 1.0f, outputSize);

    auto getIndex = [inputSize](int i, int j, int k) {
        return i * inputSize * inputSize + j * inputSize + k;
    };

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            for (int k = 0; k < outputSize; ++k) {
                float xi = new_x[i];
                float yj = new_y[j];
                float zk = new_z[k];

                int x0 = std::lower_bound(x.begin(), x.end(), xi) - x.begin() - 1;
                int y0 = std::lower_bound(y.begin(), y.end(), yj) - y.begin() - 1;
                int z0 = std::lower_bound(z.begin(), z.end(), zk) - z.begin() - 1;

                x0 = std::max(0, std::min(x0, inputSize - 2));
                y0 = std::max(0, std::min(y0, inputSize - 2));
                z0 = std::max(0, std::min(z0, inputSize - 2));

                int x1 = x0 + 1;
                int y1 = y0 + 1;
                int z1 = z0 + 1;

                float xd = (xi - x[x0]) / (x[x1] - x[x0]);
                float yd = (yj - y[y0]) / (y[y1] - y[y0]);
                float zd = (zk - z[z0]) / (z[z1] - z[z0]);

                auto lerp = [](float a, float b, float t) {
                    return a + t * (b - a);
                };

                for (int c = 0; c < 3; ++c) {
                    float c000 = gCurrentGamutArray[getIndex(x0, y0, z0)][c];
                    float c100 = gCurrentGamutArray[getIndex(x1, y0, z0)][c];
                    float c010 = gCurrentGamutArray[getIndex(x0, y1, z0)][c];
                    float c110 = gCurrentGamutArray[getIndex(x1, y1, z0)][c];
                    float c001 = gCurrentGamutArray[getIndex(x0, y0, z1)][c];
                    float c101 = gCurrentGamutArray[getIndex(x1, y0, z1)][c];
                    float c011 = gCurrentGamutArray[getIndex(x0, y1, z1)][c];
                    float c111 = gCurrentGamutArray[getIndex(x1, y1, z1)][c];

                    float c00 = lerp(c000, c100, xd);
                    float c10 = lerp(c010, c110, xd);
                    float c01 = lerp(c001, c101, xd);
                    float c11 = lerp(c011, c111, xd);

                    float c0 = lerp(c00, c10, yd);
                    float c1 = lerp(c01, c11, yd);

                    new_values[3 * (i * outputSize * outputSize + j * outputSize + k) + c] = lerp(c0, c1, zd);
                }
            }
        }
    }
}

// 17 -> 33 interpolation
void _lut_domain_interpolation(int inputSize, int outputSize) {

    int newLutSize = outputSize * outputSize * outputSize;
    interpolateLUT(final_data, inputSize, outputSize);

#ifdef DEBUG
        printf("Final data shape: %d, size: %d\n", newLutSize,outputSize);
        for (int i = 0; i < 25; ++i)
             printf("[%6d] %f %f %f\n", i, final_data[3 * i], final_data[3 * i + 1], final_data[3 * i + 2]);
        for (int i = (newLutSize-25)/2; i < newLutSize/2; ++i)
             printf("[%6d] %f %f %f\n", i, final_data[3 * i], final_data[3 * i + 1], final_data[3 * i + 2]);
        for (int i = newLutSize - 25; i < newLutSize; ++i)
             printf("[%6d] %f %f %f\n", i, final_data[3 * i], final_data[3 * i + 1], final_data[3 * i + 2]);
#endif

    for (size_t i = 0; i < newLutSize; ++i) {
        gSocOutputTable[3 * i + 0] = static_cast<uint16_t>(final_data[3 * i + 0] * 4095);
        gSocOutputTable[3 * i + 1] = static_cast<uint16_t>(final_data[3 * i + 1] * 4095);
        gSocOutputTable[3 * i + 2] = static_cast<uint16_t>(final_data[3 * i + 2] * 4095);
    }

}

//============= 4. LUT 파일 저장 =============//
void save_lut_to_file(void) {
    const char* filename = "new17Lut.txt";
    FILE* out = fopen(filename, "w");
    if(!out)
        printf( "[%s] file Error \n",__FUNCTION__);
    else {
        _make_soc_3dlut();
        fprintf(out,"[Output] LUT_STEP_SIZE = %d\n",LUT_STEP_SIZE);
        for(int i = 0; i<(LUT_STEP_SIZE * LUT_STEP_SIZE * LUT_STEP_SIZE); i++)
            fprintf(out, "[%5d] %5d, %5d, %5d\n",i,gSoc17LutTable[3*i],gSoc17LutTable[3*i+1],gSoc17LutTable[3*i+2]);
        fclose(out);
    }

    _lut_domain_interpolation(LUT_STEP_SIZE, ORIGINAL_LUT_SIZE);
    
    const char* filename2 = "new33Lut.txt";
    FILE* out2 = fopen(filename2, "w");
    if(!out2)
        printf( "[%s] file Error \n",__FUNCTION__);
    else {
        fprintf(out2,"[Output] LUT_STEP_SIZE = %d\n",ORIGINAL_LUT_SIZE);
        for(int zi = 0; zi < ORIGINAL_LUT_SIZE; ++zi) {
            for(int yi = 0; yi < ORIGINAL_LUT_SIZE; ++yi) {
                for(int xi = 0; xi < ORIGINAL_LUT_SIZE; ++xi) {
                    size_t idx = 3 * (zi * ORIGINAL_LUT_SIZE * ORIGINAL_LUT_SIZE + yi * ORIGINAL_LUT_SIZE + xi);
                    fprintf(out2, "[%5d] %5d, %5d, %5d\n", idx / 3, gSocOutputTable[idx], gSocOutputTable[idx + 1], gSocOutputTable[idx + 2]);
                }
            }
        }
        fclose(out2);
    }
#ifdef DEBUG
    printf("[SAVE] file = %s\n", filename2);
#endif
}
 
// ==================================================== Initialize =======================================================
void _make_bypass_lut(int option) {
#ifdef DEBUG
    FILE* outFile = fopen("./lutLog_bypass", "w");
    if(!outFile)
        printf( "[%s] file Error \n",__FUNCTION__);
    else
        fprintf(outFile, "Index     R,G,B Value\n");
#endif

    for (int i = 0; i < LUT_STEP_SIZE * LUT_STEP_SIZE * LUT_STEP_SIZE; i++) {
        gBypassStandardArray[i][0] = (float)(i % LUT_STEP_SIZE) / (float)(LUT_STEP_SIZE - 1);
        gBypassStandardArray[i][1] = (float)((i / LUT_STEP_SIZE) % LUT_STEP_SIZE) / (float)(LUT_STEP_SIZE - 1);
        gBypassStandardArray[i][2] = (float)((int)(i / (LUT_STEP_SIZE * LUT_STEP_SIZE)) % LUT_STEP_SIZE) / (float)(LUT_STEP_SIZE - 1);

        gCurrentGamutArray[i][0] = gBypassStandardArray[i][0];
        gCurrentGamutArray[i][1] = gBypassStandardArray[i][1];
        gCurrentGamutArray[i][2] = gBypassStandardArray[i][2];
#ifdef DEBUG
        fprintf(outFile,"[%4d] %1.5f %1.5f %1.5f\n", i, gBypassStandardArray[i][0], gBypassStandardArray[i][1], gBypassStandardArray[i][2]);
#endif
    }

#ifdef DEBUG
        if(outFile)
            fclose(outFile);
#endif
}

void _generate_graphs(void) {
    for(int i=0; i<gain_step; i++) {
        vector<GraphInfo> tmpGraph; vector<double> saturations;
        for(int k=0;k<num_saturations;k++) saturations.push_back(grid_step*(k+1));
        for(int j=0;j<num_color_angles;j++) {
            double angle = (2 * M_PI * j) / num_color_angles;
            vector<pair<int, int>> vertices;
            for(double radius: saturations)
                vertices.emplace_back(static_cast<int>(angle * FIXED_POINT_SCALE), static_cast<int>(radius * FIXED_POINT_SCALE));
            tmpGraph.emplace_back(GraphInfo(i, vertices));
        }
        graphs.emplace_back(tmpGraph);
    }
#ifdef DEBUG
    printf("[Initinalize] _generate_graphs\n");
#endif
}
void pqcontrollerCustomLut_init_custom_lut_function(void) {
    original_grpah_coordinate=GraphCoordinates(gain_step,vector<vector<pair<int,int>>>(num_color_angles,vector<pair<int,int>>(num_saturations,DEFAULT_PAIR)));
    prev_graph_coordinate=GraphCoordinates(gain_step,vector<vector<pair<int,int>>>(num_color_angles,vector<pair<int,int>>(num_saturations,DEFAULT_PAIR)));
    current_graph_coordinate=GraphCoordinates(gain_step,vector<vector<pair<int,int>>>(num_color_angles,vector<pair<int,int>>(num_saturations,DEFAULT_PAIR)));
    _generate_graphs();
    for(int i=0;i<gain_step;i++)for(int j=0;j<num_color_angles;j++)for(int k=0;k<num_saturations;k++){
        original_grpah_coordinate[i][j][k]=graphs[i][j].vertices[k];
        prev_graph_coordinate[i][j][k]=graphs[i][j].vertices[k];
        current_graph_coordinate[i][j][k]=graphs[i][j].vertices[k];
    }
    _make_bypass_lut(0);
    bIsInit = true;
#ifdef DEBUG
    printf("[Initinalize] pqcontrollerCustomLut_init_custom_lut_function\n");
#endif
}
 
//============= API & MAIN =============//
bool pqcontrollerCustomLut_isColorFineSupport(int *column, int *row, int *radius, int *gain, bool *bIsSupport, bool *bIsSetOn)
{
    *column     = num_saturations;
    *row        = num_color_angles;
    *radius     = gRadius * 100;
    *gain       = gain_step;
    *bIsSupport = true;
    *bIsSetOn   = true; //commerMenuData_getItemValueSimple(COMMER_MENU_FINE_COLOR_TUNING, FALSE);

    return true;
}

bool pqcontrollerCustomLut_setColorFineParam(vector<COLOR_TUNING_INFO_T> colorInfos, int function) {
    if(!bIsInit) { printf("Init not finish!\n"); return false; }
    apply_control_point_updates(colorInfos);
    changed_point_info = extract_changed_control_points();
    for(auto& pt : changed_point_info)
        sector_polygon_and_lut_update(pt[0], pt[1], pt[2]);
    for(auto& pt : changed_point_info)
        prev_graph_coordinate[pt[0]][pt[1]][pt[2]] = current_graph_coordinate[pt[0]][pt[1]][pt[2]];
    // Save file function.
    save_lut_to_file();
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
        _load_original_3dlut(0);
        _make_bypass_lut(0);
    }

    return true;
}

// ==================== MAIN ===========================
int main() {
    pqcontrollerCustomLut_init_custom_lut_function();
    int size = 2;
    std::string jsonData[size] = {
        R"({"gainIndex" : 16, "row" : 0, "column" : 7, "theta" : 135, "radius" : 1000})",
        R"({"gainIndex" : 14, "row" : 8, "column" : 4, "theta" : 0, "radius" : 1600})"
    };
    std::vector<COLOR_TUNING_INFO_T> colorInfoVec;
    for(int i = 0; i < size; i++) {
        COLOR_TUNING_INFO_T colorInfo;
        parseJsonString(jsonData[i], colorInfo);
#ifdef DEBUG
        printf("[%2d] [%2d,%2d,%2d] -> move point -> [%3d,%4d]\n",i,colorInfo.gainIndex,colorInfo.row,colorInfo.column,colorInfo.theta,colorInfo.radius);
#endif
        colorInfoVec.push_back(colorInfo);
    }
    pqcontrollerCustomLut_setColorFineParam(colorInfoVec, 0);
    return 0;
}