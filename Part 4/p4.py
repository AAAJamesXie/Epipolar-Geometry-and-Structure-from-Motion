import numpy as np

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    u, _, vt = np.linalg.svd(E)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    poses = []
    for basis in (w, w.T):
        rotation = u @ basis @ vt
        rotation *= np.sign(np.linalg.det(rotation))
        for sign in (1, -1):
            poses.append(np.column_stack((rotation, sign * u[:, 2])))
    return np.stack(poses)

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    rows = image_points[..., :, :, None] * camera_matrices[..., :, 2:3, :] - camera_matrices[..., :, :2, :]
    design = rows.reshape((*rows.shape[:-3], -1, 4))
    _, _, vt = np.linalg.svd(design, full_matrices=False)
    point = vt[..., -1, :]
    return point[..., :3] / point[..., 3:4]

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    projected = _project(point_3d, camera_matrices)
    residual = projected[..., :2] / projected[..., 2:3] - image_points
    return residual.reshape((*residual.shape[:-2], -1))


def _project(point_3d, camera_matrices):
    return (camera_matrices[..., :3] @ point_3d[..., None, :, None])[..., 0] + camera_matrices[..., 3]

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    projected = _project(point_3d, camera_matrices)
    depth = projected[..., 2:3]
    pixels = projected[..., :2] / depth
    derivatives = (camera_matrices[..., :2, :3] - pixels[..., None] * camera_matrices[..., 2:3, :3]) / depth[..., None]
    return derivatives.reshape((*derivatives.shape[:-3], -1, 3))

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    point = linear_estimate_3d_point(image_points, camera_matrices)
    for _ in range(10):
        residual = reprojection_error(point, image_points, camera_matrices)
        derivative = jacobian(point, camera_matrices)
        # A pseudoinverse solves the least-squares step for single or batched tracks.
        step = (np.linalg.pinv(derivative) @ residual[..., None])[..., 0]
        point -= step
    return point

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    poses = estimate_initial_RT(E)
    scores = positive_depth_counts(poses, image_points, K)
    return poses[np.argmax(scores)]


def positive_depth_counts(poses, image_points, K):
    first = np.column_stack((np.eye(3), np.zeros(3)))
    counts = []
    for pose in poses:
        cameras = K @ np.stack((first, pose))
        points = linear_estimate_3d_point(image_points, cameras)
        depth2 = points @ pose[2, :3] + pose[2, 3]
        counts.append(np.count_nonzero((points[:, 2] > 0) & (depth2 > 0)))
    return np.array(counts)

