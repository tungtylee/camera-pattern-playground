import matplotlib.pyplot as plt
import numpy as np

from utils_cam import proj, rotate_along_x, rotate_along_y, rotate_along_z

# focal length: pixel (500)
# real world: cm
height = -240
fx = 500
fy = 500


K = np.eye(3)
K[0, 0] = fx
K[1, 1] = fy
theta_to_look_first_quadrant = -45
theta_to_look_down = -90 - 75
theta_to_roll = 10

R = np.eye(3)
R = rotate_along_z(R, theta_to_look_first_quadrant)
R = rotate_along_x(R, theta_to_look_down)
R = rotate_along_z(R, theta_to_roll)

Pole = np.array([0, 0, height]).reshape(3, 1)
T = np.matmul(-R.T, Pole)

meshx, meshy = np.meshgrid(np.linspace(-300, 300, 31), np.linspace(-400, 400, 41))
X3d = meshx.reshape(-1)
Y3d = meshy.reshape(-1)
Z3d = np.zeros(X3d.shape)
x2d, y2d = proj(K, R, T, X3d, Y3d, Z3d)
X3d = np.array([0, 60, 0, 0])
Y3d = np.array([0, 0, 0, 60])
Z3d = np.zeros(X3d.shape)
x2arr, y2arr = proj(K, R, T, X3d, Y3d, Z3d)

# Plot the points
plt.figure(figsize=(8, 6))  # Set the figure size as desired
plt.scatter(x2d, y2d, c="blue", marker="o", label="Projected Points")  # Scatter plot
# x-axis
idxlist = [0, 2]
colorlist = ["red", "green"]
labellist = ["xaxis", "yaxis"]
for idx, c, label in zip(idxlist, colorlist, labellist):
    dx = x2arr[idx + 1] - x2arr[idx]
    dy = y2arr[idx + 1] - y2arr[idx]
    plt.arrow(
        x2arr[idx],
        y2arr[idx],
        dx,
        dy,
        width=2,
        head_width=12,
        head_length=12,
        linewidth=2,
        fc=c,
        ec=c,
    )
    plt.quiver(
        0,
        0,
        0,
        0,
        color=c,
        scale=1,
        label=label,
        headwidth=6,
        headlength=7,
    )


plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("2D Projection of 3D Points")
plt.legend()
plt.grid(True)  # Optional, adds a grid
plt.axis([-fx, fx, -fy, fy])
# plt.axis("equal")  # Ensures equal scaling on both axes
plt.show()
