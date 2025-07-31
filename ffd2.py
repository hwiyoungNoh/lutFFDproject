import numpy as np
import matplotlib.pyplot as plt

# Parameters
grid_size = 15
num_levels = 5
num_axes = 8
center = (grid_size - 1) / 2
max_radius = center
step = 0.2  # Step size for grid points

xx, yy = np.meshgrid(np.arange(0, grid_size, step), np.arange(0, grid_size, step))
points = np.vstack([xx.ravel(), yy.ravel()]).T

def generate_multilevel_spider_web(num_axes, num_levels, max_radius):
    angles = np.linspace(0, 2*np.pi, num_axes, endpoint=False)
    grid_points = []
    for level in range(1, num_levels+1):
        r = max_radius * level / num_levels
        ring_points = [(center + r*np.cos(a), center + r*np.sin(a)) for a in angles]
        grid_points.append(ring_points)
    return np.array(grid_points), angles

deformations = np.array([0.0, 0.3, -0.2, 0.35, 0.0, -0.1, 0.23, -0.28])
level_spacings = np.array([1.0, 1.05, 1.13, 1.2, 1.3])
level_ranges = np.linspace(0, max_radius, num_levels + 1)

def deform_point(x, y, axis_angles, deformations, level_spacings, level_ranges):
    dx, dy = x - center, y - center
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx) % (2*np.pi)
    axis_idx = np.argmin([min(abs(theta - a), 2*np.pi-abs(theta - a)) for a in axis_angles])
    deformation = deformations[axis_idx]
    level_idx = np.digitize([r], level_ranges)[0] - 1
    spacing = level_spacings[min(level_idx, len(level_spacings)-1)]
    r_new = r * (1 + deformation) * spacing
    x_new = center + r_new * np.cos(theta)
    y_new = center + r_new * np.sin(theta)
    return x_new, y_new, level_idx

grid_points, axis_angles = generate_multilevel_spider_web(num_axes, num_levels, max_radius)

original_levels = []
for pt in points:
    dx, dy = pt[0] - center, pt[1] - center
    r = np.sqrt(dx**2 + dy**2)
    level_idx = np.digitize([r], level_ranges)[0] - 1
    original_levels.append(level_idx)
original_levels = np.array(original_levels)

deformed_points = []
deformed_levels = []
for pt in points:
    x_new, y_new, level_idx = deform_point(pt[0], pt[1], axis_angles, deformations, level_spacings, level_ranges)
    deformed_points.append([x_new, y_new])
    deformed_levels.append(level_idx)
deformed_points = np.array(deformed_points)
deformed_levels = np.array(deformed_levels)

cmap = plt.cm.viridis
norm = plt.Normalize(0, num_levels - 1)

fig, axes = plt.subplots(1, 2, figsize=(24, 12))  # Enlarged figure size
# ---- [Original] ----
ax = axes[0]
sc = ax.scatter(points[:, 0], points[:, 1], c=original_levels, cmap=cmap, s=12, alpha=0.6, marker='s', edgecolor='none')
for i, ring in enumerate(grid_points):
    xs, ys = zip(*ring)
    ax.plot(xs + (xs[0],), ys + (ys[0],), ls='--', color=cmap(i/num_levels), lw=2, label=f'Level {i+1}')
for j in range(num_axes):
    xs = [grid_points[0, j, 0], grid_points[-1, j, 0]]
    ys = [grid_points[0, j, 1], grid_points[-1, j, 1]]
    ax.plot(xs, ys, color='gray', lw=1)
ax.set_title('Original 15x15 Grid & Spider Web (step=0.2)')
ax.set_aspect('equal')
ax.set_xlim(-1, grid_size)
ax.set_ylim(-1, grid_size)
ax.grid(True)

# ---- [Deformed] ----
ax = axes[1]
sc2 = ax.scatter(deformed_points[:, 0], deformed_points[:, 1], c=deformed_levels, cmap=cmap, s=12, alpha=0.6, marker='s', edgecolor='none')
for i in range(num_levels):
    ring = []
    for j in range(num_axes):
        x, y, _ = deform_point(grid_points[i, j, 0], grid_points[i, j, 1], axis_angles, deformations, level_spacings, level_ranges)
        ring.append((x, y))
    xs, ys = zip(*ring)
    ax.plot(xs + (xs[0],), ys + (ys[0],), color=cmap(i/num_levels), lw=3)
for j in range(num_axes):
    xs = []
    ys = []
    for i in range(num_levels):
        x, y, _ = deform_point(grid_points[i, j, 0], grid_points[i, j, 1], axis_angles, deformations, level_spacings, level_ranges)
        xs.append(x)
        ys.append(y)
    ax.plot(xs, ys, color='k', lw=1)
ax.set_title('FFD Deformed 15x15 Grid & Spider Web (step=0.2)')
ax.set_aspect('equal')
ax.set_xlim(-1, grid_size)
ax.set_ylim(-1, grid_size)
ax.grid(True)

fig.tight_layout()
plt.show()