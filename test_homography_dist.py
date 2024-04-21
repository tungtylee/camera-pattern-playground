import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from utils_cam import genCircle, proj, rotate_along_x, rotate_along_y, rotate_along_z


def example(theta_to_look_upward=60):
    # focal length: pixel (500)
    # real world: cm
    height = -240
    shiftx = -40
    shifty = -40
    fx = 1
    fy = 1

    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    theta_to_look_first_quadrant = -45
    theta_to_roll = 0

    R = np.eye(3)
    R = rotate_along_z(R, theta_to_look_first_quadrant)
    R = rotate_along_x(R, theta_to_look_upward)
    R = rotate_along_z(R, theta_to_roll)

    Pole = np.array([shiftx, shifty, height]).reshape(3, 1)
    T = np.matmul(R, -Pole)

    meshx, meshy = np.meshgrid(np.linspace(-300, 300, 31), np.linspace(-400, 400, 41))
    X3dgrid = meshx.reshape(-1)
    Y3dgrid = meshy.reshape(-1)
    Z3dgrid = np.zeros(X3dgrid.shape)
    x2dgrid, y2dgrid = proj(K, R, T, X3dgrid, Y3dgrid, Z3dgrid)

    X3d = np.array([0, 60, 0, 0])
    Y3d = np.array([0, 0, 0, 60])
    Z3d = np.zeros(X3d.shape)
    x2arr, y2arr = proj(K, R, T, X3d, Y3d, Z3d)

    # Plot the points
    fig, ax = plt.subplots(figsize=(8, 6))  # Set the figure size as desired
    plt.scatter(
        x2dgrid, y2dgrid, c="blue", marker="o", label="Projected Points"
    )  # Scatter plot
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
            width=0.02,
            head_width=0.12,
            head_length=0.12,
            linewidth=0.02,
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

    titleline = [
        "2D Projection of 3D Points",
        f"theta_to_look_upward: {theta_to_look_upward} (90 is vertical)",
    ]
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("\n".join(titleline))
    plt.legend()
    plt.grid(True)  # Optional, adds a grid
    plt.axis([-fx, fx, -fy, fy])
    # plt.axis("equal")  # Ensures equal scaling on both axes

    points_3d = np.stack((X3dgrid, Y3dgrid, Z3dgrid), axis=1)
    points_2d = np.stack((x2dgrid, y2dgrid), axis=1)

    ngrid = points_3d.shape[0]
    # Select 32 random indices from the range of 0 to 1999 without replacement
    random_indices = np.random.choice(ngrid, 32, replace=False)

    # Use these indices to select a subset of points
    # Only x, y coordinates for homography
    selected_points_3d = points_3d[random_indices, :2]
    selected_points_2d = points_2d[random_indices]

    # Compute the homography using RANSAC
    homography_matrix, status = cv2.findHomography(
        selected_points_3d, selected_points_2d, cv2.RANSAC
    )
    inverse_homography_matrix = np.linalg.inv(homography_matrix)

    print("Homography Matrix:")
    print(homography_matrix)
    # print(inverse_homography_matrix)
    # print("Status (Inlier = 1, Outlier = 0):")
    # print(status)

    scale = 1 / np.linalg.norm(homography_matrix[:, 0])
    t = homography_matrix[:, 2] * scale
    r1 = homography_matrix[:, 0] * scale
    r2 = homography_matrix[:, 1] * scale

    origin_in_3d = [np.dot(r1, -t), np.dot(r2, -t)]
    print(origin_in_3d)

    x2cam, y2cam = proj(K, R, T, origin_in_3d[0], origin_in_3d[1], 0)
    ax.plot(x2cam, y2cam, "yx", markersize=12)

    # Annotate the point
    ax.annotate(
        "Camera",
        (x2cam, y2cam),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

    def midpoints(vertices):
        mids = []
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            mids.append((mid_x, mid_y))
        return mids

    # 100 cm
    for dis in [100, 200, 400, 800]:
        X3dcircle, Y3dcircle = genCircle(n_points=1000, radius=dis, center=origin_in_3d)
        Z3dcircle = np.zeros(X3dcircle.shape)
        x2arr, y2arr = proj(K, R, T, X3dcircle, Y3dcircle, Z3dcircle)
        vertices = np.column_stack((x2arr, y2arr))
        mids = midpoints(vertices)
        polygon = patches.Polygon(vertices, closed=True, fill=None, edgecolor="r")
        ax.add_patch(polygon)
        nmids = len(mids)
        seplen = int(nmids / 10)
        for midx in range(0, nmids, seplen):
            mid = mids[midx]
            label = "{}m".format(dis / 100)
            ax.annotate(
                label, mid, textcoords="offset points", xytext=(0, 10), ha="center"
            )

    plt.savefig(f"example_{theta_to_look_upward}.png")


if __name__ == "__main__":
    example(0)
    example(20)
    example(60)
