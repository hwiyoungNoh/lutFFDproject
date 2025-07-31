import matplotlib.pyplot as plt
import numpy as np

# 10x10 그리드의 모든 좌표 데이터를 생성합니다.
data_points = [(x, y) for x in range(11) for y in range(11)]

# 3단위로 존재하는 제어점 데이터를 생성합니다.
control_points = [(x, y) for x in range(0, 11, 3) for y in range(0, 11, 3)]

# 이동 전의 그래프를 그립니다.
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Before Moving Control Point")
plt.xticks(range(11))
plt.yticks(range(11))
plt.grid(True)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 모든 좌표 데이터를 점으로 표시합니다.
for point in data_points:
    plt.scatter(point[0], point[1], color='red')

# 제어점을 다른 색상으로 표시합니다.
for point in control_points:
    plt.scatter(point[0], point[1], color='blue', s=100)

# 특정 제어점을 이동시킵니다.
old_control_point = (3, 3)
new_control_point = (4.5, 4.5)

# 이동 전 선택한 제어점을 강조합니다.
plt.scatter(old_control_point[0], old_control_point[1], color='green', s=200, edgecolor='black')

# FFD를 적용하여 이동 후의 좌표 데이터를 계산합니다.
def apply_ffd(data_points, control_points, old_cp, new_cp):
    new_data_points = []
    for x, y in data_points:
        # 제어점이 아닌 경우에만 이동 계산
        if (x, y) not in control_points:
            # 제어점의 인덱스를 찾습니다.
            i = x // 3
            j = y // 3
            # 제어점의 상대적 위치를 계산합니다.
            u = (x % 3) / 3.0
            v = (y % 3) / 3.0
            # 제어점의 인덱스가 경계를 벗어나지 않도록 합니다.
            if i < len(control_points) // 4 and j < len(control_points) // 4:
                # 새로운 좌표를 계산합니다.
                new_x = (1 - u) * (1 - v) * control_points[i * 4 + j][0] + \
                        u * (1 - v) * control_points[(i + 1) * 4 + j][0] + \
                        (1 - u) * v * control_points[i * 4 + (j + 1)][0] + \
                        u * v * control_points[(i + 1) * 4 + (j + 1)][0]
                new_y = (1 - u) * (1 - v) * control_points[i * 4 + j][1] + \
                        u * (1 - v) * control_points[(i + 1) * 4 + j][1] + \
                        (1 - u) * v * control_points[i * 4 + (j + 1)][1] + \
                        u * v * control_points[(i + 1) * 4 + (j + 1)][1]
                new_data_points.append((new_x, new_y))
            else:
                new_data_points.append((x, y))
        else:
            new_data_points.append((x, y))
    return new_data_points

# 제어점을 업데이트합니다.
updated_control_points = [(new_control_point if point == old_control_point else point) for point in control_points]

# 새로운 좌표 데이터를 계산합니다.
new_data_points = apply_ffd(data_points, updated_control_points, old_control_point, new_control_point)

# 이동 후의 그래프를 그립니다.
plt.subplot(1, 2, 2)
plt.title("After Moving Control Point")
plt.xticks(range(11))
plt.yticks(range(11))
plt.grid(True)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 새로운 좌표 데이터를 점으로 표시합니다.
for point in new_data_points:
    plt.scatter(point[0], point[1], color='orange')

# 새로운 제어점을 다른 색상으로 표시합니다.
for point in updated_control_points:
    if point == new_control_point:
        plt.scatter(new_control_point[0], new_control_point[1], color='green', s=200, edgecolor='black')
    else:
        plt.scatter(point[0], point[1], color='blue', s=100)

# 이동 전후의 데이터 포인트를 선으로 연결합니다.
for old_point, new_point in zip(data_points, new_data_points):
    if old_point != new_point:
        plt.plot([old_point[0], new_point[0]], [old_point[1], new_point[1]], color='gray', linestyle='--')

plt.show()