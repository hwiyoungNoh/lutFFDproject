에러발생

#include <iostream>
#include <vector>
#include <cmath>
#include <tuple> // std::tuple, std::make_tuple
#include <set> // std::set
#include <algorithm> // std::upper_bound
#include <random> // std::random_device, std::mt19937, std::uniform_real_distribution

const int grid_size = 15;
const int num_levels = 5;
const int num_axes = 8;
const int num_layers = 5;
const double center = (grid_size - 1) / 2.0;
const double max_radius = center;
const double step = 0.2;
const double sharpness = 8.0;

std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> result(num);
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

std::vector<std::vector<std::pair<double, double>>> generate_multilevel_spider_web(int num_axes, int num_levels, double max_radius) {
    std::vector<double> angles = linspace(0, 2 * M_PI, num_axes);
    std::vector<std::vector<std::pair<double, double>>> grid_points(num_levels);
    for (int level = 1; level <= num_levels; ++level) {
        double r = max_radius * level / num_levels;
        for (double angle : angles) {
            grid_points[level - 1].emplace_back(center + r * cos(angle), center + r * sin(angle));
        }
    }
    return grid_points;
}

std::vector<std::tuple<int, int, int>> get_adjacent_control_points(int z_level, int axis, int r_level) {
    std::vector<std::tuple<int, int, int>> adjacent_points;

    for (int dz = -1; dz <= 1; ++dz) {
        int adj_z_level = z_level + dz;
        if (adj_z_level < 0 || adj_z_level >= num_levels) continue;

        for (int da = -1; dz <= 1; ++da) {
            int adj_axis = (axis + da + num_axes) % num_axes;

            for (int dr = -1; dr <= 1; ++dr) {
                int adj_r_level = r_level + dr;
                if (adj_r_level < 0 || adj_r_level >= num_levels) continue;

                if (dz == 0 && da == 0 && dr == 0) continue; // Skip the selected point itself

                adjacent_points.emplace_back(adj_z_level, adj_axis, adj_r_level);
            }
        }
    }

    return adjacent_points;
}

double linear_interp_z(double z, double z_low, double z_high, double val_low, double val_high) {
    if (z_high == z_low) {
        return val_low;
    }
    double t = (z - z_low) / (z_high - z_low);
    return val_low * (1 - t) + val_high * t;
}

std::tuple<double, double, double, int> smooth_deform_point_3d_interpolated(
    double x, double y, double z, const std::vector<double>& axis_angles,
    const std::vector<std::vector<double>>& deformations_3d,
    const std::vector<std::vector<double>>& angle_deformations,
    const std::vector<double>& level_spacings, const std::vector<double>& level_ranges,
    const std::vector<double>& layer_zs, double sharpness) {

    double dx = x - center;
    double dy = y - center;
    double r = std::sqrt(dx * dx + dy * dy);
    double theta = std::atan2(dy, dx);
    if (theta < 0) theta += 2 * M_PI;

    auto it = std::upper_bound(layer_zs.begin(), layer_zs.end(), z);
    int idx_above = std::distance(layer_zs.begin(), it);
    int idx_low = std::max(0, idx_above - 1);
    int idx_high = std::min(static_cast<int>(layer_zs.size()) - 1, idx_above);

    std::vector<double> angle_diffs(axis_angles.size());
    for (size_t i = 0; i < axis_angles.size(); ++i) {
        angle_diffs[i] = std::abs(std::fmod(theta - axis_angles[i] + M_PI, 2 * M_PI) - M_PI);
    }

    std::vector<double> weights(angle_diffs.size());
    double sum_weights = 0.0;
    for (size_t i = 0; i < angle_diffs.size(); ++i) {
        weights[i] = std::exp(-sharpness * angle_diffs[i] * angle_diffs[i]);
        sum_weights += weights[i];
    }
    for (double& weight : weights) {
        weight /= sum_weights;
    }

    double deformation_low = 0.0;
    double deformation_high = 0.0;
    double angle_deformation_low = 0.0;
    double angle_deformation_high = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        deformation_low += weights[i] * deformations_3d[idx_low][i];
        deformation_high += weights[i] * deformations_3d[idx_high][i];
        angle_deformation_low += weights[i] * angle_deformations[idx_low][i];
        angle_deformation_high += weights[i] * angle_deformations[idx_high][i];
    }

    double deformation = linear_interp_z(z, layer_zs[idx_low], layer_zs[idx_high], deformation_low, deformation_high);
    double angle_deformation = linear_interp_z(z, layer_zs[idx_low], layer_zs[idx_high], angle_deformation_low, angle_deformation_high);

    int level_idx = std::distance(level_ranges.begin(), std::upper_bound(level_ranges.begin(), level_ranges.end(), r)) - 1;
    double spacing = level_spacings[std::min(level_idx, static_cast<int>(level_spacings.size()) - 1)];

    double r_new = r * (1 + deformation) * spacing;
    double theta_new = theta + angle_deformation;
    double x_new = center + r_new * std::cos(theta_new);
    double y_new = center + r_new * std::sin(theta_new);
    double z_new = z;

    return std::make_tuple(x_new, y_new, z_new, level_idx);
}

std::vector<std::vector<double>> generate_deformations(int num_layers, int num_axes) {
    std::vector<std::vector<double>> deformations_3d(num_layers, std::vector<double>(num_axes));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.04, 0.04);

    for (int i = 0; i < num_layers; ++i) {
        double phase = i / static_cast<double>(num_layers - 1) * M_PI;
        for (int j = 0; j < num_axes; ++j) {
            deformations_3d[i][j] = 0.2 * std::sin(2 * M_PI * j / num_axes + phase) + dis(gen);
        }
    }
    return deformations_3d;
}

std::vector<std::vector<double>> generate_angle_deformations(int num_layers, int num_axes) {
    std::vector<std::vector<double>> angle_deformations(num_layers, std::vector<double>(num_axes));
    for (int i = 0; i < num_layers; ++i) {
        for (int j = 0; j < num_axes; ++j) {
            angle_deformations[i][j] = 0.1 * std::sin(2 * M_PI * j / num_axes + i * M_PI / num_layers);
        }
    }
    return angle_deformations;
}

// 여러 제어점을 이동시키는 함수 (원좌표계 사용)
void move_control_points(std::vector<std::vector<std::pair<double, double>>>& grid_points, const std::vector<std::tuple<int, int, double, double>>& changes) {
    for (const auto& change : changes) {
        int level, axis;
        double dr, dtheta;
        std::tie(level, axis, dr, dtheta) = change;

        double x = grid_points[level][axis].first;
        double y = grid_points[level][axis].second;
        double dx = x - center;
        double dy = y - center;
        double r = std::sqrt(dx * dx + dy * dy);
        double theta = std::atan2(dy, dx);
        if (theta < 0) theta += 2 * M_PI;

        double r_new = r + dr;
        double theta_new = theta + dtheta;

        double x_new = center + r_new * std::cos(theta_new);
        double y_new = center + r_new * std::sin(theta_new);

        grid_points[level][axis] = std::make_pair(x_new, y_new);
    }
}

// 특정 제어점과 인접한 제어점들 사이의 영역을 계산하는 함수
std::set<int> calculate_affected_points(
    const std::vector<std::vector<double>>& points,
    const std::vector<std::vector<std::pair<double, double>>>& grid_points,
    const std::vector<std::tuple<int, int, double, double>>& changes,
    std::vector<std::tuple<int, int, int, int, int, double, double>>& adjacent_control_points) {

    std::set<int> affected_indices;

    for (const auto& change : changes) {
        int z_level, axis;
        double dr, dtheta;
        std::tie(z_level, axis, dr, dtheta) = change;

        double x = grid_points[z_level][axis].first;
        double y = grid_points[z_level][axis].second;
        double dx = x - center;
        double dy = y - center;
        double r = std::sqrt(dx * dx + dy * dy);
        double theta = std::atan2(dy, dx);
        if (theta < 0) theta += 2 * M_PI;

        double r_new = r + dr;
        double theta_new = theta + dtheta;

        double x_new = center + r_new * std::cos(theta_new);
        double y_new = center + r_new * std::sin(theta_new);

        // 인접한 제어점들 추가
        adjacent_control_points.push_back(std::make_tuple(z_level, axis, z_level, z_level, axis, z_level, r, theta));

        // 현재 레벨의 인접한 제어점들
        for (int i = -1; i <= 1; ++i) {
            int adj_z_level = z_level + i;
            if (adj_z_level < 0 || adj_z_level >= num_levels) continue;

            for (int j = -1; j <= 1; ++j) {
                int adj_axis = (axis + j + num_axes) % num_axes;
                for (int k = -1; k <= 1; ++k) {
                    int adj_r_level = z_level + k;
                    if (adj_r_level < 0 || adj_r_level >= num_levels) continue;

                    double adj_x = grid_points[adj_r_level][adj_axis].first;
                    double adj_y = grid_points[adj_r_level][adj_axis].second;
                    double adj_dx = adj_x - center;
                    double adj_dy = adj_y - center;
                    double adj_r = std::sqrt(adj_dx * adj_dx + adj_dy * adj_dy);
                    double adj_theta = std::atan2(adj_dy, adj_dx);
                    if (adj_theta < 0) adj_theta += 2 * M_PI;
                    adjacent_control_points.push_back(std::make_tuple(z_level, axis, z_level, adj_z_level, adj_axis, adj_r_level, adj_r, adj_theta));
                }
            }
        }

        for (size_t i = 0; i < points.size(); ++i) {
            double px = points[i][0];
            double py = points[i][1];
            double pz = points[i][2];

            double pdx = px - center;
            double pdy = py - center;
            double pr = std::sqrt(pdx * pdx + pdy * pdy);
            double ptheta = std::atan2(pdy, pdx);
            if (ptheta < 0) ptheta += 2 * M_PI;

            if (std::abs(pr - r_new) < step && std::abs(ptheta - theta_new) < step) {
                affected_indices.insert(i);
            }
        }
    }

    return affected_indices;
}

// 변경된 제어점의 영향을 받는 영역의 점들만 변형하는 함수
std::vector<std::vector<double>> deform_points_in_region(
    const std::vector<std::vector<double>>& points,
    const std::vector<std::vector<std::pair<double, double>>>& grid_points,
    const std::vector<double>& axis_angles,
    const std::vector<std::vector<double>>& deformations_3d,
    const std::vector<std::vector<double>>& angle_deformations,
    const std::vector<double>& level_spacings,
    const std::vector<double>& level_ranges,
    const std::vector<double>& layer_zs,
    double sharpness,
    const std::set<int>& affected_indices) {

    std::vector<std::vector<double>> deformed_points = points;

    for (int idx : affected_indices) {
        double x = points[idx][0];
        double y = points[idx][1];
        double z = points[idx][2];

        double x_new, y_new, z_new;
        int level_idx;
        std::tie(x_new, y_new, z_new, level_idx) = smooth_deform_point_3d_interpolated(
            x, y, z, axis_angles, deformations_3d,
            angle_deformations, level_spacings, level_ranges, layer_zs, sharpness);
        deformed_points[idx] = {x_new, y_new, z_new};
    }

    return deformed_points;
}

int main() {
    std::vector<double> z_vals = linspace(0, grid_size - 1, 15);
    std::vector<double> layer_zs = linspace(0, grid_size - 1, num_layers);
    std::vector<std::vector<double>> points;

    for (double z : z_vals) {
        for (double x = 0; x < grid_size; x += step) {
            for (double y = 0; y < grid_size; y += step) {
                points.push_back({x, y, z});
            }
        }
    }

    std::vector<std::vector<std::pair<double, double>>> grid_points = generate_multilevel_spider_web(num_axes, num_levels, max_radius);
    std::vector<double> axis_angles = linspace(0, 2 * M_PI, num_axes);
    std::vector<std::vector<double>> deformations_3d = generate_deformations(num_layers, num_axes);
    std::vector<std::vector<double>> angle_deformations = generate_angle_deformations(num_layers, num_axes);
    std::vector<double> level_spacings = {1.0, 1.05, 1.13, 1.2, 1.3};
    std::vector<double> level_ranges = linspace(0, max_radius, num_levels + 1);

    std::vector<std::vector<double>> deformed_points;
    std::vector<int> deformed_levels;

    // 여러 제어점을 이동시키는 예제 (원좌표계 사용)
    std::vector<std::tuple<int, int, double, double>> changes = {
        {1, 0, 2.0, M_PI / 4}, // 첫 번째 레벨의 첫 번째 축을 반지름 방향으로 2.0, 각도 방향으로 45도 이동
        {1, 1, 1.0, M_PI / 6}, // 두 번째 레벨의 두 번째 축을 반지름 방향으로 1.0, 각도 방향으로 30도 이동
        // 추가적인 제어점 이동을 여기에 추가할 수 있습니다.
    };
    move_control_points(grid_points, changes);

    // 영향을 받는 점들과 인접한 제어점들을 계산
    std::vector<std::tuple<int, int, int, int, int, double, double>> adjacent_control_points;
    std::set<int> affected_indices = calculate_affected_points(points, grid_points, changes, adjacent_control_points);

    // 영향을 받는 영역의 점들만 변형
    deformed_points = deform_points_in_region(points, grid_points, axis_angles, deformations_3d, angle_deformations, level_spacings, level_ranges, layer_zs, sharpness, affected_indices);

    // 변형된 grid 데이터를 출력
    for (const auto& pt : deformed_points) {
        std::cout << "x: " << pt[0] << ", y: " << pt[1] << ", z: " << pt[2] << std::endl;
    }

    // 선택한 제어점과 인접한 제어점들의 정보 출력
    int selected_z = 1;
    int selected_axis = 0;
    int selected_r = 1;

    std::vector<std::tuple<int, int, int>> adjacent_points = get_adjacent_control_points(selected_z, selected_axis, selected_r);

    std::cout << "Selected: (" << selected_z << "," << selected_axis << "," << selected_r << ")" << std::endl;

    std::cout << "Adjacent points:" << std::endl;
    for (const auto& point : adjacent_points) {
        int z, axis, r;
        std::tie(z, axis, r) = point;
        std::cout << "(" << z << "," << axis << "," << r << ")" << std::endl;
    }

    return 0;
}