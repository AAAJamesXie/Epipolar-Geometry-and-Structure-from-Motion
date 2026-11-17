from pathlib import Path

import numpy as np

from p4 import (
    estimate_initial_RT, estimate_RT_from_E, jacobian, linear_estimate_3d_point,
    nonlinear_estimate_3d_point, positive_depth_counts, reprojection_error,
)
from sfm_utils import Frame, bundle_adjustment, merge_all_frames

ROOT = Path(__file__).resolve().parent


def load_provided(path):
    return np.load(ROOT / "data" / path, allow_pickle=True, encoding="latin1")


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=8, suppress=True)
    focal = 719.5459
    k = np.diag([focal, focal, 1.])
    matches = load_provided("statue/matches_subset.npy")[0]
    fundamentals = load_provided("statue/fundamental_matrices.npy")[0]
    essential = k.T @ fundamentals[0] @ k
    example = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
                        [0.1019, 0.9948, 0.0045, -0.0089],
                        [0.2041, -0.0254, 0.9786, 0.0331]])
    poses = estimate_initial_RT(essential)
    pose_diff = np.max(np.abs(poses - example), axis=(1, 2))
    assert pose_diff.min() < 1e-4
    print("A nearest candidate, max abs difference:", pose_diff.argmin() + 1, pose_diff.min())
    print("A rotations and translation:", poses[0, :, :3], poses[2, :, :3], poses[0, :, 3])

    first = np.column_stack((np.eye(3), np.zeros(3)))
    cameras = k @ np.stack((first, example))
    first_match = matches[0][:, 0].reshape(2, 2)
    expected_point = np.array([0.6774, -1.1029, 4.6621])
    point = linear_estimate_3d_point(first_match, cameras)
    u, v = first_match.T
    design = np.vstack((cameras[:, 0] - u[:, None] * cameras[:, 2],
                        cameras[:, 1] - v[:, None] * cameras[:, 2]))
    _, vectors = np.linalg.eigh(design.T @ design)
    independent = vectors[:3, 0] / vectors[3, 0]
    np.testing.assert_allclose(point, independent, atol=1e-10, rtol=0)
    print("B point and reference max difference:", point, np.max(np.abs(point - expected_point)))

    error = reprojection_error(expected_point, first_match, cameras)
    expected_error = np.array([-0.0095458, -0.5171407, 0.0059307, 0.501631])
    derivative = jacobian(expected_point, cameras)
    expected_jacobian = np.array([[154.33943931, 0., -22.42541691],
                                  [0., 154.33943931, 36.51165089],
                                  [141.87950588, -14.27738422, -56.20341644],
                                  [21.9792766, 149.50628901, 32.23425643]])
    np.testing.assert_allclose(error, expected_error, atol=1e-6, rtol=0)
    np.testing.assert_allclose(derivative, expected_jacobian, atol=1e-6, rtol=0)
    print("C residual:", error)
    print("C Jacobian:", derivative)

    cameras4 = load_provided("unit_test_camera_matrix.npy")
    observations = load_provided("unit_test_image_matches.npy")
    linear = linear_estimate_3d_point(observations, cameras4)
    refined = nonlinear_estimate_3d_point(observations, cameras4)
    norms = [np.linalg.norm(reprojection_error(p, observations, cameras4)) for p in (linear, refined)]
    assert norms[1] < norms[0]
    print("D errors:", *norms)

    scores = positive_depth_counts(poses, matches[0].T.reshape(-1, 2, 2), k)
    selected = estimate_RT_from_E(essential, observations[None, :2, :], k)
    np.testing.assert_allclose(selected, example, atol=1e-4, rtol=0)
    print("E scores and selected pose:", scores, selected)

    image_paths = sorted((ROOT / "data/statue/images").glob("*.jpg"))
    height, width = plt.imread(image_paths[0]).shape[:2]
    assert len(image_paths) == len(matches) + 1 == len(fundamentals) + 1
    frames = []
    for i, (tracks, matrix) in enumerate(zip(matches, fundamentals), 1):
        frame = Frame(tracks.T, focal, matrix, width, height)
        bundle_adjustment(frame)
        frames.append(frame)
        print(f"Adjusted sparse pair {i}", flush=True)
    merged = merge_all_frames(frames)
    assert len(merged.motion) == len(image_paths)
    print("F merged cameras and sparse points:", len(merged.motion), len(merged.structure), flush=True)

    dense_matches = load_provided("statue/dense_matches.npy")
    assert len(dense_matches) == len(merged.motion) - 1
    dense = []
    for i, tracks in enumerate(dense_matches):
        projections = merged.K @ merged.motion[i:i + 2]
        points = nonlinear_estimate_3d_point(tracks.T.reshape(-1, 2, 2), projections)
        print(f"F {image_paths[i].stem}-{image_paths[i + 1].stem}: {len(points)} tracks", flush=True)
        dense.append(points)
    points = np.vstack(dense)
    finite = points[np.isfinite(points).all(axis=1)]
    assert len(finite) > 0
    print("F finite points:", len(finite), "of", len(points), flush=True)

    lower, upper = np.array([-5., -5., 0.]), np.array([5., 5., 10.])
    visible = finite[((finite >= lower) & (finite <= upper)).all(axis=1)]
    print("F points in display bounds:", len(visible), flush=True)
    assert len(visible) > 0
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(9, 8), layout="constrained")
    axes = fig.add_subplot(projection="3d")
    axes.scatter(*visible.T, c="#171717", s=1.1, depthshade=True)
    axes.set(xlim=(-5, 5), ylim=(-5, 5), zlim=(0, 10), xlabel="X", ylabel="Y", zlabel="Z")
    axes.set_box_aspect((1, 1, 1))
    axes.view_init(elev=-100, azim=90)
    fig.savefig(figures / "statue-reconstruction.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    run()

