import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
# 파라미터
grid_size = 15
num_levels = 5      # 동심원 단계
num_axes = 8        # 방사형 축 개수
num_layers = 5      # z축 레이어 (제어점 개수)
center = (grid_size - 1) / 2
max_radius = center
step = 0.2
sharpness = 8
 
# 좌표 데이터 (z방향 15단계)
z_vals = np.linspace(0, grid_size-1, 15)
layer_zs = np.linspace(0, grid_size-1, num_layers)  # 실제 제어 레이어
xx, yy = np.meshgrid(np.arange(0, grid_size, step), np.arange(0, grid_size, step))
xys = np.vstack([xx.ravel(), yy.ravel()]).T
points = np.array([[x, y, z] for z in z_vals for x, y in xys])
 
def generate_multilevel_spider_web(num_axes, num_levels, max_radius):
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False)
    grid_points = []
    for level in range(1, num_levels + 1):
        r = max_radius * level / num_levels
        ring_points = [(center + r * np.cos(a), center + r * np.sin(a)) for a in angles]
        grid_points.append(ring_points)
    return np.array(grid_points), angles
 
# 각 레이어별로 다르게 제어점 변형률 부여!
deformations_3d = []
for i in range(num_layers):
    phase = i / (num_layers-1) * np.pi  # 높이에 따라 패턴 변화
    base = 0.2 * np.sin(np.linspace(0, 2*np.pi, num_axes, endpoint=False) + phase)
    noise = 0.08 * (np.random.rand(num_axes) - 0.5)
    deformations_3d.append(base + noise)
deformations_3d = np.array(deformations_3d)
 
level_spacings = np.array([1.0, 1.05, 1.13, 1.2, 1.3])
level_ranges = np.linspace(0, max_radius, num_levels + 1)
 
def linear_interp_z(z, z_low, z_high, val_low, val_high):
    if z_high == z_low:
        return val_low
    t = (z - z_low) / (z_high - z_low)
    return val_low * (1 - t) + val_high * t
 
def smooth_deform_point_3d_interpolated(x, y, z, axis_angles, deformations_3d, level_spacings, level_ranges, layer_zs, sharpness=8):
    dx, dy = x - center, y - center
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx) % (2 * np.pi)
    idx_above = np.searchsorted(layer_zs, z)
    if idx_above == 0:
        idx_low, idx_high = 0, 0
    elif idx_above >= len(layer_zs):
        idx_low, idx_high = len(layer_zs) - 1, len(layer_zs) - 1
    else:
        idx_low, idx_high = idx_above - 1, idx_above
    angle_diffs = np.abs((theta - axis_angles + np.pi) % (2 * np.pi) - np.pi)
    weights = np.exp(-sharpness * angle_diffs ** 2)
    weights /= np.sum(weights)
    deformation_low = np.dot(weights, deformations_3d[idx_low])
    deformation_high = np.dot(weights, deformations_3d[idx_high])
    deformation = linear_interp_z(z, layer_zs[idx_low], layer_zs[idx_high], deformation_low, deformation_high)
    level_idx = np.digitize([r], level_ranges)[0] - 1
    spacing = level_spacings[min(level_idx, len(level_spacings) - 1)]
    r_new = r * (1 + deformation) * spacing
    x_new = center + r_new * np.cos(theta)
    y_new = center + r_new * np.sin(theta)
    z_new = z
    return x_new, y_new, z_new, level_idx
 
grid_points, axis_angles = generate_multilevel_spider_web(num_axes, num_levels, max_radius)
deformed_points, deformed_levels = [], []
for pt in points:
    x_new, y_new, z_new, level_idx = smooth_deform_point_3d_interpolated(
        pt[0], pt[1], pt[2], axis_angles, deformations_3d,
        level_spacings, level_ranges, layer_zs, sharpness)
    deformed_points.append([x_new, y_new, z_new])
    deformed_levels.append(level_idx)
deformed_points = np.array(deformed_points)
deformed_levels = np.array(deformed_levels)
 
# 3D 시각화: 각 z 단계별 컨트롤포인트 + 스파이더 그리드도 함께
fig = plt.figure(figsize=(22, 10))
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
 
# 1. 원본 데이터
cmap = plt.cm.viridis
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='cool', s=3, alpha=0.4)
ax.set_title('원본 3D Grid & 각 z 단계별 Spider Web')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_xlim(0, grid_size); ax.set_ylim(0, grid_size); ax.set_zlim(0, grid_size)
 
# 각 레이어별 스파이더웹 컨트롤 포인트/격자 (원본)
for i, z0 in enumerate(layer_zs):
    for l in range(num_levels):
        ring = grid_points[l]
        xs, ys = zip(*ring)
        ax.plot(xs + (xs[0],), ys + (ys[0],), [z0]*(num_axes+1), ls='--', color=cmap(l/num_levels), lw=2)
    # 제어점(가장 바깥 ring, 원본 좌표) 별도 강조
    out_ring_x, out_ring_y = zip(*grid_points[-1])
    ax.scatter(out_ring_x, out_ring_y, [z0]*num_axes, color='k', s=70, marker='o', alpha=0.7)
    ax.plot(out_ring_x + (out_ring_x[0],), out_ring_y + (out_ring_y[0],), [z0]*(num_axes+1), color='k', lw=3, alpha=0.7)
 
# 2. FFD 변형 데이터
ax2.scatter(deformed_points[:, 0], deformed_points[:, 1], deformed_points[:, 2],
            c=deformed_points[:, 2], cmap='cool', s=3, alpha=0.4)
ax2.set_title('FFD 변형 후 3D Grid\n(z 단계별 제어점, 패턴 변경율 각기 다름)')
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
ax2.set_xlim(0, grid_size); ax2.set_ylim(0, grid_size); ax2.set_zlim(0, grid_size)
 
for i, z0 in enumerate(layer_zs):
    for l in range(num_levels):
        ring = []
        for j in range(num_axes):
            # 각 레이어 z에서 마지막(가장 바깥) ring의 각 제어점 변형 위치
            x_base, y_base = grid_points[l, j]
            x, y, z__, _ = smooth_deform_point_3d_interpolated(x_base, y_base, z0, axis_angles, deformations_3d, level_spacings, level_ranges, layer_zs, sharpness)
            ring.append((x, y, z0))
        ring = np.array(ring)
        ax2.plot(np.append(ring[:,0], ring[0,0]), np.append(ring[:,1], ring[0,1]), np.append(ring[:,2], ring[0,2]), color=cmap(l/num_levels), lw=2)
    # 제어점(방사축 끝점) 강조
    out_ring = []
    for j in range(num_axes):
        x_base, y_base = grid_points[-1, j]
        x, y, z__, _ = smooth_deform_point_3d_interpolated(x_base, y_base, z0, axis_angles, deformations_3d, level_spacings, level_ranges, layer_zs, sharpness)
        out_ring.append((x, y, z0))
    out_ring = np.array(out_ring)
    ax2.scatter(out_ring[:,0], out_ring[:,1], out_ring[:,2], color='crimson', s=90, marker='o', edgecolor='k', linewidth=1.5, zorder=10, alpha=0.95)
    ax2.plot(np.append(out_ring[:,0], out_ring[0,0]), np.append(out_ring[:,1], out_ring[0,1]), np.append(out_ring[:,2], out_ring[0,2]), color='crimson', lw=3, alpha=0.8)
 
plt.tight_layout()
plt.show()