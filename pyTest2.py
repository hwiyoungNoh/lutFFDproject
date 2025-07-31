import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D grid 생성
grid_shape = (21, 21, 21)
x = np.linspace(-5, 5, grid_shape[0])
y = np.linspace(-5, 5, grid_shape[1])
z = np.linspace(-5, 5, grid_shape[2])
XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')

# 각 grid point의 좌표 배열 (N, 3)
points = np.stack([XX, YY, ZZ], axis=-1)

# 중심 포인트 및 이동 벡터 정의
center = np.array([0, 0, 0])    # 이동시킬 중심 위치
T = np.array([2, 2, 2])         # 이동 벡터

# 각 포인트마다 중심과의 거리 계산
distances = np.linalg.norm(points - center, axis=-1)

# 역수 가중치 계산
weights = 1 / (distances + 1e-6)  # 0으로 나누는 것을 방지하기 위해 작은 값을 더함
weights = np.clip(weights, 0, 1)  # 가중치가 0보다 작아지지 않도록 클리핑

# 각 grid포인트별로 이동량 적용
delta = weights[..., None] * T  # (N, 3)

# 이동 적용: 새 위치 계산
moved_points = points + delta

# 시각화
fig = plt.figure(figsize=(12, 6))

# 이동 전
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(points[..., 0].flatten(), points[..., 1].flatten(), points[..., 2].flatten(), c='b', s=1)
ax1.set_title('Before Movement')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# 이동 후
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(moved_points[..., 0].flatten(), moved_points[..., 1].flatten(), moved_points[..., 2].flatten(), c='r', s=1)
ax2.set_title('After Movement')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.show()