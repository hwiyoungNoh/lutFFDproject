#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <thread>
#include <chrono>
 
using namespace std;
 
#define LUT_STEP_SIZE 17
#define ORIGINAL_LUT_SIZE 33
#define FIXED_POINT_SCALE 100.0f
 
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
using GraphCoordinates = vector<vector<vector<pair<int, int>>>>;
const pair<int, int> DEFAULT_PAIR = {0,0};
 
vector<vector<GraphInfo>> graphs;
GraphCoordinates original_grpah_coordinate, prev_graph_coordinate, current_graph_coordinate;
vector<vector<int>> changed_point_info;
 
int num_saturations = 8; // level
int num_color_angles = 12; // axis
int grid_step = 2;
int gain_step = LUT_STEP_SIZE;
unsigned int gOriginalLutTable[LUT_STEP_SIZE * LUT_STEP_SIZE * LUT_STEP_SIZE] = {0,};
float gBypassStandardArray[LUT_STEP_SIZE * LUT_STEP_SIZE * LUT_STEP_SIZE][3] = {0,};
bool bIsInit = false;
 
//---------------------- HSV, RGB 변환 -------------------------//
void rgb_to_hsv(double r, double g, double b, double& h, double& s, double& v) {
    double M = std::max(r, std::max(g, b));
    double m = std::min(r, std::min(g, b));
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
    int Hi = int(std::floor(h*6.0));
    double f = h*6.0 - Hi;
    double p = v*(1-s), q = v*(1-s*f), t = v*(1-(s*(1-f)));
    switch (Hi%6) {
        case 0: rgb[0]=v; rgb[1]=t; rgb[2]=p; break;
        case 1: rgb[0]=q; rgb[1]=v; rgb[2]=p; break;
        case 2: rgb[0]=p; rgb[1]=v; rgb[2]=t; break;
        case 3: rgb[0]=p; rgb[1]=q; rgb[2]=v; break;
        case 4: rgb[0]=t; rgb[1]=p; rgb[2]=v; break;
        case 5: rgb[0]=v; rgb[1]=p; rgb[2]=q; break;
    }
}
 
//---------------------- sector polygon, 보간, LUT 갱신 util ------------------//
struct Ctl { int gain, axis, level; };
vector<Ctl> get_adjacent_controls(int gain, int axis, int level) {
    vector<Ctl> adj;
    for (int dg = -1; dg <= 1; ++dg) {
        int ng = gain + dg; if (ng < 0 || ng >= LUT_STEP_SIZE) continue;
        for (int da = -1; da <= 1; ++da) {
            int na = (axis + da + num_color_angles) % num_color_angles;
            for (int dl = -1; dl <= 1; ++dl) {
                int nl = level + dl; if (nl < 0 || nl >= num_saturations) continue;
                if (dg == 0 && da == 0 && dl == 0) continue;
                adj.push_back({ng,na,nl});
            }
        }
    }
    return adj;
}
pair<double,double> polar_to_cart(double h, double s) {
    double theta = h*2*M_PI, r=s; return std::make_pair(r*cos(theta), r*sin(theta));
}
bool point_in_polygon(double x, double y, const vector<pair<double,double>>& poly) {
    int n=poly.size(); bool in=false;
    for(int i=0,j=n-1;i<n;j=i++) {
        double xi=poly[i].first,yi=poly[i].second;
        double xj=poly[j].first,yj=poly[j].second;
        if((yi>y)!=(yj>y)&&
           (x<(xj-xi)*(y-yi)/((yj-yi)+1e-12)+xi)) in=!in;
    }
    return in;
}
void smooth_deform(const double h0, const double s0, const double v0,
                   const double h1, const double s1, const double v1,
                   double h, double s, double v,
                   double& hh, double& ss, double& vv)
{
    double d = sqrt((h-h1)*(h-h1)+(s-s1)*(s-s1)+(v-v1)*(v-v1));
    double alpha = exp(-d*8.0);
    hh = h*(1-alpha)+h1*alpha;
    ss = s*(1-alpha)+s1*alpha;
    vv = v*(1-alpha)+v1*alpha;
}
void find_nearest_lut(int gain, int axis, int level, int& r_base, int& g_base, int& b_base) {
    double th = (double)graphs[gain][axis].vertices[level].first / FIXED_POINT_SCALE;
    double tr = (double)graphs[gain][axis].vertices[level].second / FIXED_POINT_SCALE;
    double h0 = th/(2*M_PI), s0 = tr/(grid_step * num_saturations), v0 = (gain+1)/double(LUT_STEP_SIZE);
    double min_dist = std::numeric_limits<double>::max();
    int rr=0,gg=0,bb=0;
    for(int r=0;r<LUT_STEP_SIZE;r++)
    for(int g=0;g<LUT_STEP_SIZE;g++)
    for(int b=0;b<LUT_STEP_SIZE;b++) {
        double hh,ss,vv;
        rgb_to_hsv(r/16.0,g/16.0,b/16.0,hh,ss,vv);
        double d = (hh-h0)*(hh-h0)+(ss-s0)*(ss-s0)+(vv-v0)*(vv-v0);
        if(d<min_dist) { min_dist=d; rr=r; gg=g; bb=b; }
    }
    r_base=rr; g_base=gg; b_base=bb;
}
 
void sector_polygon_and_lut_update(int gain, int axis, int level,
                                   int gain_new, int axis_new, int level_new)
{
    vector<Ctl> adj_ctls = get_adjacent_controls(gain, axis, level);
    vector<pair<double,double>> poly;
    auto ctl_to_hs = [](int g, int a, int l) {
        double th = (double)graphs[g][a].vertices[l].first / FIXED_POINT_SCALE;
        double tr = (double)graphs[g][a].vertices[l].second / FIXED_POINT_SCALE;
        double h = th/(2*M_PI), s = tr/(grid_step * num_saturations);
        return polar_to_cart(h,s);
    };
    poly.push_back(ctl_to_hs(gain,axis,level));
    for(auto ctl: adj_ctls) poly.push_back(ctl_to_hs(ctl.gain,ctl.axis,ctl.level));
    vector<int> all_levels; all_levels.push_back(level); for(auto c: adj_ctls) all_levels.push_back(c.level);
    int min_lv = *std::min_element(all_levels.begin(), all_levels.end());
    int max_lv = *std::max_element(all_levels.begin(), all_levels.end());
 
    vector<vector<vector<bool>>> sector_mask(LUT_STEP_SIZE,vector<vector<bool>>(LUT_STEP_SIZE,vector<bool>(LUT_STEP_SIZE,false)));
    for(int r=0;r<LUT_STEP_SIZE;r++)
    for(int g=0;g<LUT_STEP_SIZE;g++)
    for(int b=0;b<LUT_STEP_SIZE;b++) {
        double hh,ss,vv;
        rgb_to_hsv(r/16.0, g/16.0, b/16.0, hh, ss, vv);
        int grid_lv = std::round(vv*(num_saturations-1));
        pair<double,double> cart = polar_to_cart(hh,ss);
        if(grid_lv<min_lv || grid_lv>max_lv) continue;
        if(point_in_polygon(cart.first, cart.second, poly)) sector_mask[r][g][b]=true;
    }
 
    int r_base,g_base,b_base;
    find_nearest_lut(gain_new, axis_new, level_new, r_base, g_base, b_base);
    unsigned int base_val = gOriginalLutTable[(r_base*LUT_STEP_SIZE+g_base)*LUT_STEP_SIZE+b_base];
    double th1 = (double)graphs[gain_new][axis_new].vertices[level_new].first / FIXED_POINT_SCALE;
    double tr1 = (double)graphs[gain_new][axis_new].vertices[level_new].second / FIXED_POINT_SCALE;
    double h1 = th1/(2*M_PI), s1 = tr1/(grid_step*num_saturations), v1 = (gain_new+1)/double(LUT_STEP_SIZE);
 
    printf("선택 제어점: gain=%d axis=%d level=%d, 이동후(gain=%d axis=%d level=%d)\n"
           "  -> base RGB: (%d %d %d)\n", gain, axis, level, gain_new, axis_new, level_new, r_base, g_base, b_base);
 
    // 이동/보간 및 LUT 적용, 디버그 정보
    int out_cnt = 0;
    for(int r=0;r<LUT_STEP_SIZE;r++)
    for(int g=0;g<LUT_STEP_SIZE;g++)
    for(int b=0;b<LUT_STEP_SIZE;b++) {
        if(!sector_mask[r][g][b]) continue;
        double hh,ss,vv;
        rgb_to_hsv(r/16.0, g/16.0, b/16.0, hh, ss, vv);
        double hh2,ss2,vv2;
        smooth_deform(hh, ss, vv, h1, s1, v1, hh2, ss2, vv2);
 
        double rgb_new[3], hsv_new[3] = {hh2, ss2, vv2};
        hsv_to_rgb(hsv_new, rgb_new);
        int rr = std::round(rgb_new[0]*16), gg = std::round(rgb_new[1]*16), bb = std::round(rgb_new[2]*16);
        rr = std::min(std::max(rr,0),16); gg = std::min(std::max(gg,0),16); bb = std::min(std::max(bb,0),16);
        gOriginalLutTable[(r*LUT_STEP_SIZE+g)*LUT_STEP_SIZE+b] = ((rr&0xFF)<<16)|((gg&0xFF)<<8)|(bb&0xFF);
 
        // 디버깅 출력
        if(out_cnt++ < 50)
            printf("  grid: [%2d %2d %2d], old HSV: %.2f %.2f %.2f → new HSV: %.2f %.2f %.2f → new RGB: %2d %2d %2d\n",
                    r,g,b,hh,ss,vv,hh2,ss2,vv2,rr,gg,bb);
        else if(out_cnt == 51) printf("  ... 이하 생략 ...\n");
    }
}
 
void save_lut_to_file(const char* filename) {
    FILE* out = fopen(filename, "w");
    for(int r=0;r<LUT_STEP_SIZE;r++)
    for(int g=0;g<LUT_STEP_SIZE;g++)
    for(int b=0;b<LUT_STEP_SIZE;b++) {
        unsigned int v = gOriginalLutTable[(r*LUT_STEP_SIZE+g)*LUT_STEP_SIZE+b];
        fprintf(out, "%3d %3d %3d\n", (v>>16)&0xFF, (v>>8)&0xFF, v&0xFF);
    }
    fclose(out);
}
void sleep_millisec(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
 
//-----------------------------------
// API/json 파싱/테스트 구간
//-----------------------------------
void parseJsonString(const std::string& jsonData, COLOR_TUNING_INFO_T& colorInfo) {
    std::istringstream ss(jsonData);
    string key, value;
    while (ss >> key) {
        if (key.find("gainIndex") != string::npos) { ss.ignore(3); ss >> colorInfo.gainIndex; }
        else if (key.find("row") != string::npos) { ss.ignore(3); ss >> colorInfo.row; }
        else if (key.find("column") != string::npos) { ss.ignore(3); ss >> colorInfo.column; }
        else if (key.find("theta") != string::npos) { ss.ignore(3); ss >> colorInfo.theta; }
        else if (key.find("radius") != string::npos) { ss.ignore(3); ss >> colorInfo.radius; }
    }
}
bool pqcontrollerCustomLut_setColorFineParam(vector<COLOR_TUNING_INFO_T> colorInfo, int function)
{
    if(bIsInit != true) { printf("Init not finish!\n"); return false; }
    for( auto a : colorInfo) {
        graphs[a.gainIndex][a.row].vertices[a.column] = {a.theta, a.radius};
        current_graph_coordinate[a.gainIndex][a.row][a.column] = {a.theta, a.radius};
    }
    changed_point_info.clear();
    for (auto a : colorInfo) { changed_point_info.push_back({a.gainIndex, a.row, a.column}); }
    for (auto a : colorInfo) {
        sector_polygon_and_lut_update(a.gainIndex, a.row, a.column, a.gainIndex, a.row, a.column);
    }
    save_lut_to_file("newLut.txt");
    return true;
}
 
// ... 그래프/좌표 초기화, LUT 초기화 등 init 코드 추가 필요 (생략) ...

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

    /*
    if (function == 5) {
        _make_bypass_lut(0);
    }
    */

    return true;
}


int main() {
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