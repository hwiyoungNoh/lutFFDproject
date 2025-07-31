#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <tuple>

int steps_count = 33;

// HSV to RGB conversion function
void HSVtoRGB(float H, float S, float V, float &R, float &G, float &B) {
    float C = V * S;
    float X = C * (1 - fabs(fmod(H / 60.0, 2) - 1));
    float m = V - C;
    float r, g, b;

    if (0 <= H && H < 60) {
        r = C, g = X, b = 0;
    } else if (60 <= H && H < 120) {
        r = X, g = C, b = 0;
    } else if (120 <= H && H < 180) {
        r = 0, g = C, b = X;
    } else if (180 <= H && H < 240) {
        r = 0, g = X, b = C;
    } else if (240 <= H && H < 300) {
        r = X, g = 0, b = C;
    } else {
        r = C, g = 0, b = X;
    }

    R = r + m;
    G = g + m;
    B = b + m;
}

// Function to calculate the difference between two RGB values
std::tuple<int, int, int> calculateRGBDifference(int R1, int G1, int B1, int R2, int G2, int B2) {
    return std::make_tuple(abs(R1 - R2), abs(G1 - G2), abs(B1 - B2));
}

// Function to find the closest step in 33 steps for a given value
int findClosestStep(float value) {
    float step_size = 1.0f / (steps_count - 1);
    int closest_step = round(value / step_size);
    return closest_step;
}

// Function to find all steps between two values
std::vector<int> findStepsBetween(float start, float end) {
    std::vector<int> steps;
    float step_size = 1.0f / (steps_count - 1);

    float min_value = std::min(start, end);
    float max_value = std::max(start, end);

    for (int i = 0; i < steps_count; ++i) {
        float step_value = i * step_size;
        if (step_value >= min_value && step_value <= max_value) {
            steps.push_back(i);
        }
    }

    return steps;
}

int main() {
    const int H_steps = 12;
    const int S_steps = 8;
    const int V_steps = 16;

    // Select a specific HSV step
    int selected_h = 5; // Example: 5th step for H
    int selected_s = 3; // Example: 3rd step for S
    int selected_v = 10; // Example: 10th step for V

    float selected_H = (selected_h * 360.0f) / H_steps;
    float selected_S = selected_s / (float)(S_steps - 1);
    float selected_V = selected_v / (float)(V_steps - 1);

    float selected_R, selected_G, selected_B;
    HSVtoRGB(selected_H, selected_S, selected_V, selected_R, selected_G, selected_B);

    // Find the closest steps for the selected RGB values
    int closest_R = findClosestStep(selected_R);
    int closest_G = findClosestStep(selected_G);
    int closest_B = findClosestStep(selected_B);

    std::cout << std::left << std::setw(15) << "Selected HSV:" 
              << "H: " << std::setw(10) << selected_H 
              << "S: " << std::setw(10) << selected_S 
              << "V: " << std::setw(10) << selected_V 
              << " -> R: " << std::setw(10) << closest_R 
              << "G: " << std::setw(10) << closest_G 
              << "B: " << std::setw(10) << closest_B << std::endl;

    // Calculate RGB differences with adjacent HSV steps
    std::vector<std::tuple<int, int, int>> differences;

    for (int dh = -1; dh <= 1; ++dh) {
        for (int ds = -1; ds <= 1; ++ds) {
            for (int dv = -1; dv <= 1; ++dv) {
                if (dh == 0 && ds == 0 && dv == 0) continue; // Skip the selected step itself

                int adj_h = selected_h + dh;
                int adj_s = selected_s + ds;
                int adj_v = selected_v + dv;

                // Ensure the adjacent steps are within valid range
                if (adj_h < 0 || adj_h >= H_steps || adj_s < 0 || adj_s >= S_steps || adj_v < 0 || adj_v >= V_steps) {
                    continue;
                }

                float adj_H = (adj_h * 360.0f) / H_steps;
                float adj_S = adj_s / (float)(S_steps - 1);
                float adj_V = adj_v / (float)(V_steps - 1);

                float adj_R, adj_G, adj_B;
                HSVtoRGB(adj_H, adj_S, adj_V, adj_R, adj_G, adj_B);

                // Find the closest steps for the adjacent RGB values
                int closest_adj_R = findClosestStep(adj_R);
                int closest_adj_G = findClosestStep(adj_G);
                int closest_adj_B = findClosestStep(adj_B);

                auto diff = calculateRGBDifference(closest_R, closest_G, closest_B, closest_adj_R, closest_adj_G, closest_adj_B);
                differences.push_back(diff);

                std::cout << std::left << std::setw(15) << "Adjacent HSV:" 
                          << "H: " << std::setw(10) << adj_H 
                          << "S: " << std::setw(10) << adj_S 
                          << "V: " << std::setw(10) << adj_V 
                          << " -> R: " << std::setw(10) << closest_adj_R 
                          << "G: " << std::setw(10) << closest_adj_G 
                          << "B: " << std::setw(10) << closest_adj_B 
                          << " | Difference: R: " << std::setw(10) << std::get<0>(diff) 
                          << "G: " << std::setw(10) << std::get<1>(diff) 
                          << "B: " << std::setw(10) << std::get<2>(diff) << std::endl;

                // Find all steps between selected and adjacent RGB values
                auto R_steps = findStepsBetween(selected_R, adj_R);
                auto G_steps = findStepsBetween(selected_G, adj_G);
                auto B_steps = findStepsBetween(selected_B, adj_B);

                std::cout << "Included RGB steps between selected and adjacent HSV:" << std::endl;
                for (int r_step : R_steps) {
                    for (int g_step : G_steps) {
                        for (int b_step : B_steps) {
                            std::cout << "R: " << std::setw(10) << r_step 
                                      << "G: " << std::setw(10) << g_step 
                                      << "B: " << std::setw(10) << b_step << std::endl;
                        }
                    }
                }

                // 추가: diff가 0인 경우에도 다른 조합을 생성하여 표시
                if (std::get<0>(diff) == 0 || std::get<1>(diff) == 0 || std::get<2>(diff) == 0) {
                    std::cout << "Additional RGB steps for diff = 0:" << std::endl;
                    for (int r_step : R_steps) {
                        for (int g_step : G_steps) {
                            for (int b_step : B_steps) {
                                if (std::get<0>(diff) == 0) {
                                    r_step = closest_R;
                                }
                                if (std::get<1>(diff) == 0) {
                                    g_step = closest_G;
                                }
                                if (std::get<2>(diff) == 0) {
                                    b_step = closest_B;
                                }
                                std::cout << "R: " << std::setw(10) << r_step 
                                          << "G: " << std::setw(10) << g_step 
                                          << "B: " << std::setw(10) << b_step << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}