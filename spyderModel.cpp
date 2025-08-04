#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <tuple>
#include <set>
#include <algorithm>
#include <random>

const int grid_size = 15;
const int num_levels = 5;
const int num_axes = 8;
const int num_layers = 5; // 반드시 필요!
const double center = (grid_size - 1) / 2.0;
const double max_radius = center;
const double step = 0.2;
const double sharpness = 8.0;

// 1. linspace 구간 생성
std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> result(num);
    double st = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) result[i] = start + i * st;
    return result;
}

// 2. spider web grid 생성
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

// 3. 인접 제어점 공식(z±1, axis±1 순환, level±1, 자기자신 제외)
std::vector<std::tuple<int, int, int>> get_adjacent_control_points(int z, int axis, int level) {
    std::vector<std::tuple<int, int, int>> adj;
    for (int dz = -1; dz <= 1; ++dz) {
        int nz = z + dz;
        if (nz < 0 || nz >= num_levels) continue;
        for (int da = -1; da <= 1; ++da) {
            int naxis = (axis + da + num_axes) % num_axes;
            for (int dl = -1; dl <= 1; ++dl) {
                int nlevel = level + dl;
                if (nlevel < 0 || nlevel >= num_levels) continue;
                if (dz == 0 && da == 0 && dl == 0) continue;
                adj.emplace_back(nz, naxis, nlevel);
            }
        }
    }
    return adj;
}

// 4. 보간 interpolation
double linear_interp_z(double z, double z_low, double z_high, double val_low, double val_high) {
    if (z_high == z_low) return val_low;
    double t = (z - z_low) / (z_high - z_low);
    return val_low * (1 - t) + val_high * t;
}

// 5. 3D 부드러운 변형 보간 (axis, z 모두에 대해)
std::tuple<double, double, double, int> smooth_deform_point_3d_interpolated(
    double x, double y, double z, const std::vector<double>& axis_angles,
    const std::vector<std::vector<double>>& deformations_3d,
    const std::vector<std::vector<double>>& angle_deformations,
    const std::vector<double>& level_spacings, const std::vector<double>& level_ranges,
    const std::vector<double>& layer_zs, double sharpness)
{
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
    for (size_t i = 0; i < axis_angles.size(); ++i)
        angle_diffs[i] = std::abs(std::fmod(theta - axis_angles[i] + M_PI, 2 * M_PI) - M_PI);

    std::vector<double> weights(axis_angles.size());
    double sum_weights = 0.0;
    for (size_t i = 0; i < angle_diffs.size(); ++i) {
        weights[i] = std::exp(-sharpness * angle_diffs[i] * angle_diffs[i]);
        sum_weights += weights[i];
    }
    for (double& weight : weights) weight /= sum_weights;

    double deformation_low = 0.0, deformation_high = 0.0, angle_deformation_low = 0.0, angle_deformation_high = 0.0;
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

std::pair<double, double> polar_to_cartesian(double r, double theta) {
    return {r * std::cos(theta), r * std::sin(theta)};
}

bool point_in_polygon(double x, double y, const std::vector<std::pair<double, double>>& polygon) {
    int n = polygon.size();
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        double xi = polygon[i].first, yi = polygon[i].second;
        double xj = polygon[j].first, yj = polygon[j].second;
        bool intersect = ((yi > y) != (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

// sector polygon+z 전체 범위 후보 grid 추출 (인접점들 커버 범위 확장)
std::set<int> calculate_affected_points(
    const std::vector<std::vector<double>>& points,
    const std::vector<std::vector<std::pair<double, double>>>& grid_points,
    int selected_z, int selected_axis, int selected_level
) {
    std::set<int> affected_indices;
    auto adj_points = get_adjacent_control_points(selected_z, selected_axis, selected_level);

    // 모든 인접 레벨 포함
    std::vector<int> all_levels; all_levels.push_back(selected_level);
    for (const auto& pt : adj_points) {
        int az, aaxis, alv; std::tie(az, aaxis, alv) = pt;
        all_levels.push_back(alv);
    }
    int min_level = *std::min_element(all_levels.begin(), all_levels.end());
    int max_level = *std::max_element(all_levels.begin(), all_levels.end());
    int z_stride = grid_size / num_levels;
    int z_min = min_level * z_stride;
    int z_max = (max_level == num_levels - 1) ? grid_size - 1 : (max_level + 1) * z_stride - 1;

    // polygon 생성
    std::vector<std::tuple<int, int, int>> sector_pts = adj_points;
    sector_pts.insert(sector_pts.begin(), std::make_tuple(selected_z, selected_axis, selected_level));
    std::vector<std::pair<double, double>> polygon_coords;
    for (auto pt : sector_pts) {
        int z, axis, level; std::tie(z, axis, level) = pt;
        double x = grid_points[level][axis].first, y = grid_points[level][axis].second;
        double dx = x - center, dy = y - center;
        double r = std::sqrt(dx*dx + dy*dy);
        double theta = std::atan2(dy, dx); if (theta < 0) theta += 2*M_PI;
        polygon_coords.push_back(polar_to_cartesian(r, theta));
    }
    // grid 후보 선정
    for (size_t i = 0; i < points.size(); ++i) {
        double px = points[i][0], py = points[i][1], pz = points[i][2];
        int pz_int = static_cast<int>(round(pz));
        if (pz_int < z_min || pz_int > z_max) continue;
        double xrel = px - center, yrel = py - center;
        if (point_in_polygon(xrel, yrel, polygon_coords)) affected_indices.insert(i);
    }
    return affected_indices;
}

int main() {
    std::vector<double> z_vals = linspace(0, grid_size-1, grid_size);
    std::vector<double> layer_zs = linspace(0, grid_size-1, num_levels);
    std::vector<std::vector<double>> points;
    for (double z: z_vals)
        for (double x = 0; x < grid_size; x += step)
            for (double y = 0; y < grid_size; y += step)
                points.push_back({x, y, z});

    std::vector<std::vector<std::pair<double, double>>> grid_points = generate_multilevel_spider_web(num_axes, num_levels, max_radius);

    std::vector<double> axis_angles = linspace(0, 2 * M_PI, num_axes);

    std::vector<std::vector<double>> deformations_3d(num_layers, std::vector<double>(num_axes, 0.01));
    std::vector<std::vector<double>> angle_deformations(num_layers, std::vector<double>(num_axes, 0.005));
    std::vector<double> level_spacings = {1., 1.05, 1.13, 1.2, 1.3};
    std::vector<double> level_ranges = linspace(0, max_radius, num_levels + 1);

    std::vector<std::vector<double>> deformed_points = points;

    std::vector<std::tuple<int, int, int, int, int, int>> moves = {
        {0, 0, 0, 4, 7, 4},
        {4, 7, 4, 0, 0, 0},
        {2, 4, 2, 1, 6, 3},
        {1, 1, 1, 3, 3, 3}, // 추가 예제
        {3, 5, 2, 2, 2, 1}  // 추가 예제
    };

    for (size_t ex = 0; ex < moves.size(); ++ex) {
        int src_z, src_axis, src_level, dst_z, dst_axis, dst_level;
        std::tie(src_z, src_axis, src_level, dst_z, dst_axis, dst_level) = moves[ex];

        // 이동전 제어점 정보
        double x_src = grid_points[src_level][src_axis].first;
        double y_src = grid_points[src_level][src_axis].second;
        double r_src = std::sqrt((x_src - center)*(x_src - center) + (y_src - center)*(y_src - center));
        double theta_src = std::atan2(y_src - center, x_src - center);
        if (theta_src < 0) theta_src += 2 * M_PI; double theta_src_deg = theta_src * 180. / M_PI;

        // 이동후 제어점 정보
        double x_dst = grid_points[dst_level][dst_axis].first;
        double y_dst = grid_points[dst_level][dst_axis].second;
        double r_dst = std::sqrt((x_dst - center)*(x_dst - center) + (y_dst - center)*(y_dst - center));
        double theta_dst = std::atan2(y_dst - center, x_dst - center);
        if (theta_dst < 0) theta_dst += 2 * M_PI; double theta_dst_deg = theta_dst * 180. / M_PI;

        std::cout << "\n[예제 " << (ex + 1) << "] 제어점 이동\n";
        std::cout << " (이동 전) z=" << src_z << ", axis=" << src_axis << ", level=" << src_level
            << ", x=" << x_src << ", y=" << y_src
            << ", θ(rad):" << theta_src << ", θ(도):" << theta_src_deg << ", r:" << r_src << "\n";
        std::cout << " (이동 후) z=" << dst_z << ", axis=" << dst_axis << ", level=" << dst_level
            << ", x=" << x_dst << ", y=" << y_dst
            << ", θ(rad):" << theta_dst << ", θ(도):" << theta_dst_deg << ", r:" << r_dst << "\n";

        // 이동 전 인접 제어점 정보
        auto adj_points = get_adjacent_control_points(src_z, src_axis, src_level);
        std::cout << "인접 제어점들 (이동 전 기준, 최대 26개):\n";
        std::vector<double> layer_zs5 = linspace(0, grid_size-1, num_levels);
        for (auto& pt : adj_points) {
            int az, aax, alv; std::tie(az, aax, alv) = pt;
            double ax = grid_points[alv][aax].first;
            double ay = grid_points[alv][aax].second;
            double ar = std::sqrt((ax - center)*(ax - center) + (ay - center)*(ay - center));
            double atheta = std::atan2(ay-center, ax-center); if (atheta < 0) atheta += 2*M_PI;
            double atheta_deg = atheta * 180. / M_PI;
            std::cout << "  (" << az << ", " << aax << ", " << alv << "), z좌표:" << layer_zs5[az]
                << ", θ(rad):" << atheta << ", θ(도):" << atheta_deg << ", r:" << ar << "\n";
        }

        // sector 내부 grid 후보 (모든 z 범위 포함)
        std::set<int> affected_indices = calculate_affected_points(
            deformed_points, grid_points, src_z, src_axis, src_level);

        // 제어점 실제 이동
        grid_points[dst_level][dst_axis] = grid_points[src_level][src_axis];

        // 변형 적용(예시로 전/후 좌표만 출력, 실제 변형 함수 적용은 아래 루프에서)
        std::vector<std::vector<double>> before_move = deformed_points;
        for (int idx : affected_indices) {
            double x = before_move[idx][0], y = before_move[idx][1], z = before_move[idx][2];
            double x_new, y_new, z_new; int level_idx;
            std::tie(x_new, y_new, z_new, level_idx) = smooth_deform_point_3d_interpolated(
                x, y, z, axis_angles, deformations_3d, angle_deformations,
                level_spacings, level_ranges, layer_zs, sharpness);
            deformed_points[idx] = {x_new, y_new, z_new};
        }

        std::cout << "\n[sector 영향 grid 데이터] (x, y, z[원본])  →  (x', y', z')\n";
        std::cout << std::setw(8) << "x" << std::setw(8) << "y" << std::setw(10) << "z[원]"
                  << "   |   " << std::setw(8) << "x'" << std::setw(8) << "y'" << std::setw(8) << "z'" << "\n";
        std::cout << std::string(8 * 2 + 10 + 7 + 8 * 3, '-') << "\n";
        int show_count = 0;
        for (int idx : affected_indices) {
            if (show_count++ > 80) break; // 표 샘플만 표시 (출력이 너무 길어질 경우)
            std::cout << std::setw(8) << std::setprecision(3) << before_move[idx][0]
                << std::setw(8) << std::setprecision(3) << before_move[idx][1]
                << std::setw(10) << std::setprecision(3) << points[idx][2]
                << "   |   "
                << std::setw(8) << std::setprecision(3) << deformed_points[idx][0]
                << std::setw(8) << std::setprecision(3) << deformed_points[idx][1]
                << std::setw(8) << std::setprecision(3) << deformed_points[idx][2]
                << "\n";
        }
        if (affected_indices.size() > show_count)
            std::cout << "...(" << affected_indices.size() - show_count << " more omitted)\n";
    }

    return 0;
}
