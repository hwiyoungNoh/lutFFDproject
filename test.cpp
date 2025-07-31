#include <iostream>
#include <cmath>
#include <vector>
#include <tuple>
#include <limits>
#include <iomanip> // 출력을 일정한 간격으로 표시하기 위해 추가
#include <set> // 포함된 65단계 인덱스를 저장하기 위해 추가

using namespace std;

// 두 HSV 포인트 간의 유클리드 거리 계산 함수
double calculateDistance(int h1, int s1, int v1, int h2, int s2, int v2) {
    return sqrt(pow(h1 - h2, 2) + pow(s1 - s2, 2) + pow(v1 - v2, 2));
}

// 가장 가까운 12단계, 8단계, 16단계 인덱스를 찾는 함수
tuple<int, int, int> findClosestIndexIn12_8_16(int h, int s, int v) {
    double minDistance = numeric_limits<double>::max();
    tuple<int, int, int> closestPoint;

    for (int h12 = 0; h12 < 12; ++h12) {
        for (int s8 = 0; s8 < 8; ++s8) {
            for (int v16 = 0; v16 < 16; ++v16) {
                double distance = calculateDistance(h, s, v, h12, s8, v16);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestPoint = make_tuple(h12, s8, v16);
                }
            }
        }
    }

    return closestPoint;
}

// 인접한 단계들의 12단계, 8단계, 16단계 인덱스를 찾는 함수
vector<tuple<int, int, int>> findAdjacentIndicesIn12_8_16(int h, int s, int v) {
    vector<tuple<int, int, int>> adjacentPoints;
    for (int dh = -1; dh <= 1; ++dh) {
        for (int ds = -1; ds <= 1; ++ds) {
            for (int dv = -1; dv <= 1; ++dv) {
                if (dh == 0 && ds == 0 && dv == 0) continue; // 현재 포인트는 제외
                int nh = h + dh;
                int ns = s + ds;
                int nv = v + dv;
                if (nh >= 0 && nh < 12 && ns >= 0 && ns < 8 && nv >= 0 && nv < 16) {
                    adjacentPoints.emplace_back(nh, ns, nv);
                }
            }
        }
    }
    return adjacentPoints;
}

// 12단계, 8단계, 16단계 인덱스를 65단계 인덱스로 변환하는 함수
tuple<int, int, int> convertTo65Index(int h12, int s8, int v16) {
    int h65 = h12 * 65 / 12;
    int s65 = s8 * 65 / 8;
    int v65 = v16 * 65 / 16;
    return make_tuple(h65, s65, v65);
}

int main() {
    // 임의의 포인트 (H, S, V)를 지정
    int h = 5;  // 예시 값 (0\~11)
    int s = 3;  // 예시 값 (0\~7)
    int v = 10; // 예시 값 (0\~15)

    // 가장 가까운 12단계, 8단계, 16단계 인덱스를 찾음
    auto closestPoint12_8_16 = findClosestIndexIn12_8_16(h, s, v);

    // 결과 출력
    int closestH12, closestS8, closestV16;
    tie(closestH12, closestS8, closestV16) = closestPoint12_8_16;
    cout << "가장 가까운 12단계, 8단계, 16단계 인덱스: H=" << closestH12 << ", S=" << closestS8 << ", V=" << closestV16 << endl;

    // 인접한 단계들의 12단계, 8단계, 16단계 인덱스를 찾음
    auto adjacentPoints12_8_16 = findAdjacentIndicesIn12_8_16(closestH12, closestS8, closestV16);

    // 인접한 단계들의 결과 출력
    cout << "인접한 단계들의 12단계, 8단계, 16단계 인덱스 및 65단계 인덱스:" << endl;
    cout << setw(15) << "12단계 인덱스" << setw(25) << "65단계 인덱스" << endl;
    cout << setw(5) << "H" << setw(5) << "S" << setw(5) << "V" << setw(10) << "H" << setw(10) << "S" << setw(10) << "V" << endl;

    // 포함된 65단계 인덱스를 저장할 집합
    set<tuple<int, int, int>> included65Indices;

    for (const auto& point : adjacentPoints12_8_16) {
        int adjH12, adjS8, adjV16;
        tie(adjH12, adjS8, adjV16) = point;
        auto adjPoint65 = convertTo65Index(adjH12, adjS8, adjV16);
        int adjH65, adjS65, adjV65;
        tie(adjH65, adjS65, adjV65) = adjPoint65;
        cout << setw(5) << adjH12 << setw(5) << adjS8 << setw(5) << adjV16;
        cout << setw(10) << adjH65 << setw(10) << adjS65 << setw(10) << adjV65 << endl;
        included65Indices.insert(adjPoint65);
    }

    // 12단계, 8단계, 16단계 인덱스를 65단계 인덱스로 변환
    auto closestPoint65 = convertTo65Index(closestH12, closestS8, closestV16);

    // 결과 출력
    int closestH65, closestS65, closestV65;
    tie(closestH65, closestS65, closestV65) = closestPoint65;
    cout << "가장 가까운 65단계 인덱스: H=" << closestH65 << ", S=" << closestS65 << ", V=" << closestV65 << endl;

    // 포함된 65단계 인덱스 출력
    cout << "포함된 65단계 인덱스들:" << endl;
    cout << setw(5) << "H" << setw(5) << "S" << setw(5) << "V" << endl;
    for (const auto& index : included65Indices) {
        int h65, s65, v65;
        tie(h65, s65, v65) = index;
        cout << setw(5) << h65 << setw(5) << s65 << setw(5) << v65 << endl;
    }

    return 0;
}