import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
 
def draw_connections(ax, control_points, connections, alpha=0.3, color='gray', linewidth=0.5):
    """제어점들 간의 연결선을 그리는 함수"""
    for connection in connections:
        start_point = control_points[connection[0]]
        end_point = control_points[connection[1]]
        ax.plot([start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]],
                color=color, alpha=alpha, linewidth=linewidth)
 
# 3D grid (20×20×20) 생성
x, y, z = np.arange(20), np.arange(20), np.arange(20)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
 
# 12×8×5 control points 배치
cp_dim = (12, 8, 5)
cp_x = np.linspace(0, 19, cp_dim[0])
cp_y = np.linspace(0, 19, cp_dim[1])
cp_z = np.linspace(0, 19, cp_dim[2])
CP_X, CP_Y, CP_Z = np.meshgrid(cp_x, cp_y, cp_z, indexing='ij')
control_points = np.vstack((CP_X.ravel(), CP_Y.ravel(), CP_Z.ravel())).T
 
# 제어점들 간의 연결 관계 생성 (격자 구조)
connections = []
for i in range(cp_dim[0]):
    for j in range(cp_dim[1]):
        for k in range(cp_dim[2]):
            current_idx = i * cp_dim[1] * cp_dim[2] + j * cp_dim[2] + k
            
            # x축 방향 연결
            if i < cp_dim[0] - 1:
                next_idx = (i + 1) * cp_dim[1] * cp_dim[2] + j * cp_dim[2] + k
                connections.append([current_idx, next_idx])
            
            # y축 방향 연결
            if j < cp_dim[1] - 1:
                next_idx = i * cp_dim[1] * cp_dim[2] + (j + 1) * cp_dim[2] + k
                connections.append([current_idx, next_idx])
            
            # z축 방향 연결
            if k < cp_dim[2] - 1:
                next_idx = i * cp_dim[1] * cp_dim[2] + j * cp_dim[2] + (k + 1)
                connections.append([current_idx, next_idx])
 
connections = np.array(connections)
 
# 중앙 control point 선택 및 move delta 설정
selected_cp_idx = (cp_dim[0]//2) * cp_dim[1] * cp_dim[2] + (cp_dim[1]//2) * cp_dim[2] + (cp_dim[2]//2)
move_vector = np.array([2, -1, 1])
 
# neighbors: x/y/z 1칸 인접한 control points 포함
neighbors = []
sel_x, sel_y, sel_z = np.unravel_index(selected_cp_idx, cp_dim)
for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            nx, ny, nz = sel_x + dx, sel_y + dy, sel_z + dz
            if 0 <= nx < cp_dim[0] and 0 <= ny < cp_dim[1] and 0 <= nz < cp_dim[2]:
                neighbors.append(nx * cp_dim[1] * cp_dim[2] + ny * cp_dim[2] + nz)
neighbors = np.array(neighbors)
 
# 이동 후 control point 위치
moved_control_points = control_points.copy()
moved_control_points[neighbors] += move_vector
 
# 각 grid point가 가장 가까운 control point의 인덱스
kdtree = cKDTree(control_points)
_, indices_before = kdtree.query(points)
 
# 영향받는 point만 이동 반영
points_moved = points.copy()
for i, cp_idx in enumerate(indices_before):
    if cp_idx in neighbors:
        points_moved[i] += move_vector
 
# ------ 제어점 연결선을 포함한 3D 시각화 ------
plt.figure(figsize=(20, 10))
 
# 이동 전 그래프
ax1 = plt.subplot(121, projection='3d')
# Grid points
ax1.scatter(points[:,0], points[:,1], points[:,2], c='lightgray', alpha=0.05, s=0.5, label='Grid Points')
# 제어점들 간의 연결선
draw_connections(ax1, control_points, connections, alpha=0.4, color='lightblue', linewidth=0.8)
# 모든 제어점
ax1.scatter(control_points[:,0], control_points[:,1], control_points[:,2], c='blue', alpha=0.7, s=20, label='Control Points')
# 선택된 제어점과 그 주변
ax1.scatter(control_points[neighbors][:,0], control_points[neighbors][:,1], control_points[neighbors][:,2], c='orange', s=40, alpha=0.9, label='Affected CPs')
ax1.scatter(control_points[selected_cp_idx][0], control_points[selected_cp_idx][1], control_points[selected_cp_idx][2], c='red', s=80, label='Selected CP')
 
ax1.set_title('Before Move - Control Points with Grid Connections', fontsize=14)
ax1.set_xlim(-1, 22)
ax1.set_ylim(-1, 22)
ax1.set_zlim(-1, 22)
ax1.legend()
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
 
# 이동 후 그래프
ax2 = plt.subplot(122, projection='3d')
# Grid points
ax2.scatter(points_moved[:,0], points_moved[:,1], points_moved[:,2], c='lightgray', alpha=0.05, s=0.5, label='Moved Grid Points')
# 제어점들 간의 연결선 (이동 후)
draw_connections(ax2, moved_control_points, connections, alpha=0.4, color='lightblue', linewidth=0.8)
# 모든 제어점
ax2.scatter(moved_control_points[:,0], moved_control_points[:,1], moved_control_points[:,2], c='blue', alpha=0.7, s=20, label='Control Points')
# 이동한 제어점들
ax2.scatter(moved_control_points[neighbors][:,0], moved_control_points[neighbors][:,1], moved_control_points[neighbors][:,2], c='orange', s=40, alpha=0.9, label='Moved CPs')
ax2.scatter(moved_control_points[selected_cp_idx][0], moved_control_points[selected_cp_idx][1], moved_control_points[selected_cp_idx][2], c='red', s=80, label='Selected CP')
 
ax2.set_title('After Move - Control Points with Grid Connections', fontsize=14)
ax2.set_xlim(-1, 22)
ax2.set_ylim(-1, 22)
ax2.set_zlim(-1, 22)
ax2.legend()
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
 
plt.tight_layout()
plt.show()
 
print("FFD 제어점 연결선 시각화 완료!")
print(f"- 총 제어점: {len(control_points)}개")
print(f"- 총 연결선: {len(connections)}개")
print(f"- 이동한 제어점: {len(neighbors)}개")
print(f"- 격자 크기: 20×20×20")
print(f"- 제어점 배치: {cp_dim}")