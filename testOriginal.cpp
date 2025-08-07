#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iomanip>

float gBypassStandardArray[LUT_STEP_SIZE * LUT_STEP_SIZE * LUT_STEP_SIZE][3] = {0,};
// 다각형 정보와 선택된 꼭짓점 정보를 저장할 클래스
class PolygonInfo {
public:
    int index;
    std::vector<std::pair<double, double>> vertices;
    int selected_vertex_index;

    PolygonInfo(int idx, std::vector<std::pair<double, double>> verts)
        : index(idx), vertices(verts), selected_vertex_index(-1) {}

    void update_vertex(int vertex_index, std::pair<double, double> new_position) {
        vertices[vertex_index] = new_position;
    }

    void select_vertex(int vertex_index) {
        selected_vertex_index = vertex_index;
    }

    void deselect_vertex() {
        selected_vertex_index = -1;
    }
};

class ChangeVal {
public:
    std::string rgbColor;
    int index;
    int value;
    bool init;
};

void saveLUTfile(const std::vector<std::vector<std::vector<double>>>& array) {
    std::ofstream file("./makeLUT.CUBE");
    if (file.is_open()) {
        file << "#LUT size\n";
        file << "LUT_3D_SIZE 17\n\n";
        for (const auto& plane : array) {
            for (const auto& row : plane) {
                for (const auto& val : row) {
                    file << std::fixed << std::setprecision(6) << val << "\t";
                }
                file << "\n";
            }
        }
        file.close();
    }
}

std::pair<double, double> to_cartesian(double r, double theta) {
    return { r * std::cos(theta), r * std::sin(theta) };
}

std::pair<double, double> to_polar(double x, double y) {
    double r = std::hypot(x, y);
    double theta = std::atan2(y, x);
    return { r, theta };
}

void rgb_to_hsv(double r, double g, double b, double& h, double& s, double& v) {
    double M = std::max(r, std::max(g, b));
    double m = std::min(r, std::min(g, b));
    v = M;
    double c = M - m;
    s = (v > 0) ? c / v : 0;
    h = 0;
    if (c != 0) {
        if (M == r)      h = fmod((g - b) / c, 6.0) / 6.0;
        else if (M == g) h = ((b - r) / c + 2.0) / 6.0;
        else             h = ((r - g) / c + 4.0) / 6.0;
    }
    if (h < 0) h += 1.0;
}

void hsv_to_rgb(const double hsv[3], double rgb[3]) {
    double h = hsv[0], s = hsv[1], v = hsv[2];
    int Hi = int(floor(h * 6.0));
    double f = h * 6.0 - Hi, p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - (s * (1 - f)));
    switch (Hi % 6) {
        case 0: rgb[0] = v; rgb[1] = t; rgb[2] = p; break;
        case 1: rgb[0] = q; rgb[1] = v; rgb[2] = p; break;
        case 2: rgb[0] = p; rgb[1] = v; rgb[2] = t; break;
        case 3: rgb[0] = p; rgb[1] = q; rgb[2] = v; break;
        case 4: rgb[0] = t; rgb[1] = p; rgb[2] = v; break;
        case 5: rgb[0] = v; rgb[1] = p; rgb[2] = q; break;
    }
}

double clamp(double val, double min_val, double max_val) {
    return std::max(min_val, std::min(max_val, val));
}

std::vector<std::vector<std::vector<double>>> create_hsv_background(int xlim, int ylim, double volume) {
    int background_size = 400;
    std::vector<std::vector<std::vector<double>>> rgb(background_size, std::vector<std::vector<double>>(background_size, std::vector<double>(3, 0.0)));

    for (int i = 0; i < background_size; ++i) {
        for (int j = 0; j < background_size; ++j) {
            double x = (i / static_cast<double>(background_size)) * (xlim * 2) - xlim;
            double y = (j / static_cast<double>(background_size)) * (ylim * 2) - ylim;
            double r = std::sqrt(x * x + y * y);
            double theta = std::atan2(y, x);
            double h = (theta + M_PI) / (2 * M_PI);
            h = std::fmod(h + 0.5, 1.0);
            double s = clamp(r / std::max(xlim, ylim), 0.0, 1.0);
            double v = clamp(volume + (1 - volume) * (1 - r / std::max(xlim, ylim)), 0.0, 1.0);
            double hsv[3] = { h, s, v };
            double rgb_val[3];
            hsv_to_rgb(hsv, rgb_val);
            rgb[i][j][0] = rgb_val[0];
            rgb[i][j][1] = rgb_val[1];
            rgb[i][j][2] = rgb_val[2];
        }
    }
    return rgb;
}

void generate_polygons(std::vector<PolygonInfo>& polygons, int num_polygons, int num_vertices) {
    for (int i = 0; i < num_polygons; ++i) {
        double radius = 2 * (i + 1);
        std::vector<std::pair<double, double>> vertices;
        for (int j = 0; j < num_vertices; ++j) {
            double angle = (2 * M_PI * j) / num_vertices;
            vertices.push_back({ radius, angle });
        }
        polygons.push_back(PolygonInfo(i, vertices));
    }
}

std::vector<int> find_close_lut(double theta, double r, double v, double radius, int LUTSIZE) {
    if (r > radius) {
        r = radius;
    }

    double h = (theta + M_PI) / (2 * M_PI);
    h = std::fmod(h + 0.5, 1.0);

    double hsv[3] = { h, r / radius, v };
    double rgb[3];
    hsv_to_rgb(hsv, rgb);

    std::vector<int> rgbIndex(3);
    for (int i = 0; i < 3; ++i) {
        int base = static_cast<int>(rgb[i] / (1.0 / (LUTSIZE - 1)));
        if ((fmod(rgb[i], (1.0 / (LUTSIZE - 1)))) / (1.0 / (LUTSIZE - 1)) >= 0.5) {
            base += 1;
        }
        rgbIndex[i] = base;
    }
    return rgbIndex;
}

std::vector<double> find_lut_hsv(const std::vector<int>& lutVal, int LUTSIZE) {
    std::vector<double> rgb(3);
    for (int i = 0; i < 3; ++i) {
        rgb[i] = static_cast<double>(lutVal[i]) / (LUTSIZE - 1);
    }

    double hsv[3];
    rgb_to_hsv(rgb[0], rgb[1], rgb[2], hsv[0], hsv[1], hsv[2]);
    return { hsv[0], hsv[1], hsv[2] };
}

int getLinearArrayIndex(int r, int g, int b, int size) {
    return (b * size * size) + (g * size) + r;
}

void domain_change(const std::vector<int>& near, const std::vector<int>& ori, const std::vector<int>& target, std::vector<std::vector<std::vector<double>>>& lutArray, const std::vector<std::vector<std::vector<double>>>& originalLut, int LUTSIZE) {
    std::vector<int> oDiff(3), tDiff(3);
    for (int i = 0; i < 3; ++i) {
        oDiff[i] = near[i] - ori[i];
        tDiff[i] = near[i] - target[i];
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

        int arrayTarget = getLinearArrayIndex(tLut[0], tLut[1], tLut[2], LUTSIZE);
        int originalIdx = getLinearArrayIndex(oLutList[idx][0], oLutList[idx][1], oLutList[idx][2], LUTSIZE);
        lutArray[originalIdx] = originalLut[arrayTarget];
    }
}

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

void update_coordinate(int selected_vertice, std::vector<PolygonInfo>& polygons, std::vector<std::vector<std::vector<double>>>& lutArray, const std::vector<std::vector<std::vector<double>>>& originalLut, int LUTSIZE, double volume, int num_polygons, int num_vertices) {
    for (auto& polygon_info : polygons) {
        auto vertex = polygon_info.vertices[selected_vertice];
        update_lut_gain(polygon_info.index, selected_vertice, vertex.first, vertex.second, polygons, lutArray, originalLut, LUTSIZE, volume, num_polygons, num_vertices);
    }
}
void _make_bypass_lut(int option) {
#ifdef DEBUG
    FILE* outFile = fopen("/var/log/lutLog_bypass", "w");
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
int main() {
    int num_polygons = 8;
    int num_vertices = 12;
    double volume = 0.5;
    int LUTSIZE = 17;
    std::vector<PolygonInfo> polygons;
    generate_polygons(polygons, num_polygons, num_vertices);

    // HSV 배경 생성
    auto hsv_background = create_hsv_background(17, 17, volume);

    // LUT 데이터 초기화
    std::vector<std::vector<std::vector<double>>> lutArray(LUTSIZE, std::vector<std::vector<double>>(LUTSIZE, std::vector<double>(LUTSIZE, 0.0)));
    std::vector<std::vector<std::vector<double>>> originalLut = lutArray;

    // LUT 파일 저장
    saveLUTfile(lutArray);

    // 다각형 꼭짓점 업데이트
    update_coordinate(0, polygons, lutArray, originalLut, LUTSIZE, volume, num_polygons, num_vertices);

    return 0;
}