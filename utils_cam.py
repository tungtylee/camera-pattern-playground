import numpy as np


def genCircle(n_points, radius, center):
    # Generate points
    angles = np.linspace(
        0, 2 * np.pi, n_points, endpoint=False
    )  # Create angles evenly spaced around the circle
    # X coordinates
    x = radius * np.cos(angles) + center[0]
    # Y coordinates
    y = radius * np.sin(angles) + center[1]
    return x, y


def rotate_along_x(R, theta):
    theta = np.radians(theta)

    R2 = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    return np.matmul(R2, R)


def rotate_along_y(R, theta):
    theta = np.radians(theta)

    R2 = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    return np.matmul(R2, R)


def rotate_along_z(R, theta):
    theta = np.radians(theta)

    R2 = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return np.matmul(R2, R)


def proj(K, R, T, X3d, Y3d, Z3d):
    # Stack 3D coordinates into a single array (N x 3)
    points_3d = np.vstack((X3d, Y3d, Z3d)).T

    # Convert T to a column vector if it is not already
    if T.ndim == 1:
        T = T[:, np.newaxis]

    # Create the extrinsic matrix by combining R and T
    extrinsic_matrix = np.hstack((R, T))

    # Convert 3D points to homogeneous coordinates (add a row of 1s)
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Project points to camera coordinates
    camera_coords = points_3d_homogeneous.dot(extrinsic_matrix.T)

    # Apply the intrinsic matrix to get image coordinates in homogeneous form
    image_points_homogeneous = camera_coords.dot(K.T)

    # Convert from homogeneous coordinates to 2D coordinates
    x2d = image_points_homogeneous[:, 0] / image_points_homogeneous[:, 2]
    y2d = image_points_homogeneous[:, 1] / image_points_homogeneous[:, 2]

    fornan = image_points_homogeneous[:, 2] < 0
    x2d[fornan] = np.nan
    y2d[fornan] = np.nan

    return x2d, y2d
