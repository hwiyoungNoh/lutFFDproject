
/*
#include <iostream>
#include <iomanip>
#include <cmath>

// HSV to RGB conversion function
void HSVtoRGB(float h, float s, float v, int &r, int &g, int &b) {
    float c = v * s;
    float x = c * (1 - fabs(fmod(h / 60.0, 2) - 1));
    float m = v - c;
    float r1, g1, b1;

    if (h >= 0 && h < 60) {
        r1 = c, g1 = x, b1 = 0;
    } else if (h >= 60 && h < 120) {
        r1 = x, g1 = c, b1 = 0;
    } else if (h >= 120 && h < 180) {
        r1 = 0, g1 = c, b1 = x;
    } else if (h >= 180 && h < 240) {
        r1 = 0, g1 = x, b1 = c;
    } else if (h >= 240 && h < 300) {
        r1 = x, g1 = 0, b1 = c;
    } else {
        r1 = c, g1 = 0, b1 = x;
    }

    r = (r1 + m) * 255;
    g = (g1 + m) * 255;
    b = (b1 + m) * 255;
}

void printRGB(int h, int s, int v) {
    float hue = h * 30.0; // H ranges from 0 to 330 degrees
    float saturation = s / 7.0; // S ranges from 0 to 1
    float value = v / 16.0; // V ranges from 0 to 1

    int r, g, b;
    HSVtoRGB(hue, saturation, value, r, g, b);

    // Normalize RGB values to 0-32 range
    r = round(r / 255.0 * 32);
    g = round(g / 255.0 * 32);
    b = round(b / 255.0 * 32);

    std::cout << std::left << std::setw(5) << h << std::setw(5) << s << std::setw(5) << v
              << " -> " << std::setw(5) << r << std::setw(5) << g << std::setw(5) << b << std::endl;
}

void printAdjacentHSV(int h, int s, int v) {
    for (int dh = -1; dh <= 1; ++dh) {
        for (int ds = -1; ds <= 1; ++ds) {
            for (int dv = -1; dv <= 1; ++dv) {
                int nh = (h + dh + 12) % 12; // Handle circular nature of H
                int ns = s + ds;
                int nv = v + dv;

                // Ensure the values are within the valid range
                if (ns >= 0 && ns <= 7 && nv >= 0 && nv <= 16) {
                    printRGB(nh, ns, nv);
                }
            }
        }
    }
}

int main() {
    // Example 1
    std::cout << "Example 1:" << std::endl;
    std::cout << "H     S     V     -> R     G     B" << std::endl;
    printRGB(2, 5, 10);
    std::cout << "Adjacent HSV values and their RGB values:" << std::endl;
    printAdjacentHSV(2, 5, 10);

    // Example 2
    std::cout << "\nExample 2:" << std::endl;
    std::cout << "H     S     V     -> R     G     B" << std::endl;
    printRGB(6, 3, 8);
    std::cout << "Adjacent HSV values and their RGB values:" << std::endl;
    printAdjacentHSV(6, 3, 8);

    // Example 3
    std::cout << "\nExample 3:" << std::endl;
    std::cout << "H     S     V     -> R     G     B" << std::endl;
    printRGB(10, 7, 16);
    std::cout << "Adjacent HSV values and their RGB values:" << std::endl;
    printAdjacentHSV(10, 7, 16);

    return 0;
}


*/

#include <iostream>
#include <cmath>
#include <vector>
#include <tuple>

// HSV to RGB conversion function
void HSVtoRGB(float H, float S, float V, int &R, int &G, int &B) {
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

    R = (r + m) * 255;
    G = (g + m) * 255;
    B = (b + m) * 255;
}

// Function to calculate the difference between two RGB values
std::tuple<int, int, int> calculateRGBDifference(int R1, int G1, int B1, int R2, int G2, int B2) {
    return std::make_tuple(abs(R1 - R2), abs(G1 - G2), abs(B1 - B2));
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

    int selected_R, selected_G, selected_B;
    HSVtoRGB(selected_H, selected_S, selected_V, selected_R, selected_G, selected_B);

    std::cout << "Selected HSV: H: " << selected_H << " S: " << selected_S << " V: " << selected_V
              << " -> R: " << selected_R << " G: " << selected_G << " B: " << selected_B << std::endl;

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

                int adj_R, adj_G, adj_B;
                HSVtoRGB(adj_H, adj_S, adj_V, adj_R, adj_G, adj_B);

                auto diff = calculateRGBDifference(selected_R, selected_G, selected_B, adj_R, adj_G, adj_B);
                differences.push_back(diff);

                std::cout << "Adjacent HSV: H: " << adj_H << " S: " << adj_S << " V: " << adj_V
                          << " -> R: " << adj_R << " G: " << adj_G << " B: " << adj_B
                          << " | Difference: R: " << std::get<0>(diff) << " G: " << std::get<1>(diff) << " B: " << std::get<2>(diff) << std::endl;
            }
        }
    }

    return 0;
}