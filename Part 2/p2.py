from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    left_vectors, _, _ = np.linalg.svd(F)
    epipole = left_vectors[:, -1]
    if abs(epipole[2]) > np.finfo(float).eps:
        return epipole / epipole[2]
    return epipole

'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im2 - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    height, width = im.shape[:2]
    cx, cy = width / 2, height / 2
    translate = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    x, y, w = translate @ e

    # Choose the rotation closer to the identity.
    f = np.copysign(np.hypot(x, y), x)
    if f == 0:
        raise ValueError("The epipole coincides with the image center.")
    cos, sin = x / f, y / f
    rotate = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
    project = np.array([[1, 0, 0], [0, 1, 0], [-w / f, 0, 1]])
    return np.linalg.inv(translate) @ project @ rotate @ translate

'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    h2 = compute_H(e2, im2)
    x, y, w = e2
    cross = np.array([[0, -w, y], [w, 0, -x], [-y, x, 0]])
    mapping = cross @ F + np.outer(e2, np.ones(3))

    base = h2 @ mapping
    rectified1 = transform_points(points1, base)
    rectified2 = transform_points(points2, h2)
    coefficients, _, _, _ = np.linalg.lstsq(rectified1, rectified2[:, 0], rcond=None)
    affine = np.eye(3)
    affine[0] = coefficients
    return affine @ base, h2


def transform_points(points, homography):
    projected = points @ homography.T
    return projected / projected[:, 2:3]


def save_alignment(path, views, correspondences, offsets, title):
    """Draw both rasters on a shared y-axis, accounting for their crop offsets."""
    gap = 36
    shifts = []
    figure, axes = plt.subplots(figsize=(11, 6))
    left = 0
    for image, (offset_x, offset_y) in zip(views, offsets):
        height, width = image.shape[:2]
        axes.imshow(
            image, cmap="gray", vmin=0, vmax=255,
            extent=(left - 0.5, left + width - 0.5,
                    offset_y + height - 0.5, offset_y - 0.5),
        )
        shifts.append(left - offset_x)
        left += width + gap
    for i in np.linspace(0, len(correspondences[0]) - 1, 12, dtype=int):
        first, second = (points[i] for points in correspondences)
        axes.plot(
            [first[0] + shifts[0], second[0] + shifts[1]],
            [first[1], second[1]],
            color="#ffad32", marker="o", markersize=3, linewidth=0.9,
        )
    axes.set_xlim(-0.5, left - gap - 0.5)
    axes.set_ylim(
        max(off[1] + im.shape[0] for im, off in zip(views, offsets)) - 0.5,
        min(off[1] for off in offsets) - 0.5,
    )
    axes.set_aspect("equal")
    axes.set_title(title, loc="left")
    axes.axis("off")
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    part_dir = Path(__file__).resolve().parent
    part1 = part_dir.parent / "Part 1"
    sys.path.insert(0, str(part1))
    from epipolar_utils import compute_rectified_image, get_data_from_txt_file
    from p1 import normalized_eight_point_alg

    data = part1 / "data" / "set1"
    im1, im2 = (plt.imread(data / f"image{i}.jpg") for i in (1, 2))
    points1, points2 = (get_data_from_txt_file(data / f"pt_2D_{i}.txt") for i in (1, 2))
    F = normalized_eight_point_alg(points1, points2)
    e1, e2 = compute_epipole(F.T), compute_epipole(F)
    h1, h2 = compute_matching_homographies(e2, F, im2, points1, points2)
    projected1, projected2 = transform_points(points1, h1), transform_points(points2, h2)
    raw_gap = np.abs(points1[:, 1] - points2[:, 1])
    rectified_gap = np.abs(projected1[:, 1] - projected2[:, 1])

    np.set_printoptions(precision=8)
    print("Point pairs:", len(points1))
    print("Epipole 1:", e1)
    print("Epipole 2:", e2)
    print("Relative null-space residuals:",
          np.linalg.norm(F @ e1) / (np.linalg.norm(F) * np.linalg.norm(e1)),
          np.linalg.norm(F.T @ e2) / (np.linalg.norm(F) * np.linalg.norm(e2)))
    moved = h2 @ e2
    print("H2 @ e2 up to scale:", moved / moved[0])
    print("H1:\n", h1 / h1[2, 2])
    print("H2:\n", h2 / h2[2, 2])
    print(f"Before: mean={raw_gap.mean():.4f} px, max={raw_gap.max():.4f} px")
    print(f"After:  mean={rectified_gap.mean():.4f} px, max={rectified_gap.max():.4f} px")

    rectified1, offset1 = compute_rectified_image(im1, h1)
    rectified2, offset2 = compute_rectified_image(im2, h2)
    figures = part_dir / "figures"
    figures.mkdir(exist_ok=True)
    for i, image in enumerate((rectified1, rectified2), 1):
        plt.imsave(figures / f"rectified-{i}.png", image, cmap="gray", vmin=0, vmax=255)
    save_alignment(figures / "original-alignment.png", (im1, im2),
                   (points1, points2), ((0, 0), (0, 0)), "Original views")
    save_alignment(figures / "rectified-alignment.png", (rectified1, rectified2),
                   (projected1, projected2), (offset1, offset2), "Rectified views")


if __name__ == "__main__":
    main()
