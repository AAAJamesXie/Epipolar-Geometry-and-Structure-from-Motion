"""Two-view affine structure from motion."""
import numpy as np


def _measurement_svd(points_im1, points_im2):
    coordinates = np.vstack((points_im1[:, :2].T, points_im2[:, :2].T))
    centered = coordinates - coordinates.mean(axis=1, keepdims=True)
    return np.linalg.svd(centered, full_matrices=False)


def factorization_method(points_im1, points_im2):
    """Return a 3-by-N structure matrix and a 4-by-3 motion matrix."""
    left, singular_values, right = _measurement_svd(points_im1, points_im2)
    weights = np.sqrt(singular_values[:3])
    motion = left[:, :3] * weights
    structure = weights[:, None] * right[:3]
    return structure, motion


def alignment_errors(structure, truth):
    """Fit similarity and affine maps using the known 3D correspondences."""
    source = structure.T
    centered_source = source - source.mean(axis=0)
    centered_truth = truth - truth.mean(axis=0)

    # SVD leaves the handedness of the recovered coordinates arbitrary.
    left, singular_values, right = np.linalg.svd(centered_source.T @ centered_truth)
    scale = singular_values.sum() / np.sum(centered_source ** 2)
    similarity = scale * centered_source @ (left @ right) + truth.mean(axis=0)
    affine_map, _, _, _ = np.linalg.lstsq(centered_source, centered_truth, rcond=None)
    affine = centered_source @ affine_map + truth.mean(axis=0)
    return similarity, affine


def save_comparison(path, truth, aligned):
    """Overlay corresponding point clouds from two angles on equal axes."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(10.4, 4.8), layout="constrained")
    all_points = np.vstack((truth, aligned))
    minima, maxima = all_points.min(axis=0), all_points.max(axis=0)
    centers = (minima + maxima) / 2
    radius = (maxima - minima).max() * 0.54
    for index, azimuth in enumerate((-60, 35), start=1):
        axes = figure.add_subplot(1, 2, index, projection="3d")
        axes.view_init(elev=23, azim=azimuth)
        axes.scatter(*truth.T, marker="o", facecolors="none", edgecolors="#1355a1",
                     s=47, depthshade=False, label="Ground truth")
        axes.scatter(*aligned.T, marker="x", color="#c64b15", s=32,
                     depthshade=False, label="Reconstruction")
        for setter, center in zip((axes.set_xlim, axes.set_ylim, axes.set_zlim), centers):
            setter(center - radius, center + radius)
        axes.set_box_aspect((1, 1, 1))
        axes.set(xlabel="X", ylabel="Y", zlabel="Z", title=f"Azimuth {azimuth}°")
        if index == 1:
            axes.legend(loc="upper left", fontsize=9)
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)


def main():
    from pathlib import Path
    import sys

    import matplotlib
    matplotlib.use("Agg")
    root = Path(__file__).resolve().parent
    part1 = root.parent / "Part 1"
    sys.path.insert(0, str(part1))
    from epipolar_utils import get_data_from_txt_file

    def read_set(name, image_points):
        truth = get_data_from_txt_file(root / "data" / name / "pt_3D.txt")
        points = tuple(get_data_from_txt_file(image_points / f"pt_2D_{i}.txt") for i in (1, 2))
        return points, truth

    def rmse(prediction, truth):
        return np.sqrt(np.mean(np.sum((prediction - truth) ** 2, axis=1)))

    def evaluate(label, structure, truth):
        similarity, affine = alignment_errors(structure, truth)
        radius = rmse(truth, truth.mean(axis=0))
        print(f"{label} 3D RMSE (similarity, affine):",
              f"{rmse(similarity, truth):.6f}", f"{rmse(affine, truth):.6f}")
        print(f"{label} normalized RMSE (%):",
              f"{100 * rmse(similarity, truth) / radius:.4f}",
              f"{100 * rmse(affine, truth) / radius:.4f}")
        return similarity

    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    full_points, full_truth = read_set("set1", part1 / "data" / "set1")
    full_structure, _ = factorization_method(*full_points)
    aligned = evaluate("Full set", full_structure, full_truth)
    save_comparison(figures / "full-reconstruction.png", full_truth, aligned)
    full_values = _measurement_svd(*full_points)[1]
    print("Full singular values:", *(f"{value:.8f}" for value in full_values))
    print(f"Full fourth-to-third ratio: {100 * full_values[3] / full_values[2]:.4f}%")

    subset_points, subset_truth = read_set("set1_subset", root / "data" / "set1_subset")
    subset_structure, _ = factorization_method(*subset_points)
    aligned = evaluate("Subset", subset_structure, subset_truth)
    save_comparison(figures / "subset-reconstruction.png", subset_truth, aligned)

    # Match the subset by exact 3D coordinates and verify both image tracks.
    index = {tuple(point): i for i, point in enumerate(full_truth)}
    shared = [index[tuple(point)] for point in subset_truth]
    assert len(set(shared)) == len(shared)
    for full_view, subset_view in zip(full_points, subset_points):
        np.testing.assert_array_equal(full_view[shared], subset_view)
    evaluate("Full fit on subset", full_structure[:, shared], subset_truth)
    subset_values = _measurement_svd(*subset_points)[1]
    print("Subset singular values:", *(f"{value:.8f}" for value in subset_values))
    print(f"Subset fourth-to-third ratio: {100 * subset_values[3] / subset_values[2]:.4f}%")


if __name__ == "__main__":
    main()

