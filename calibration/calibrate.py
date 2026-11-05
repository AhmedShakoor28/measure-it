"""Calibrate the phone camera from checkerboard photos and save the result."""
import glob
import sys

import cv2
import numpy as np

PATTERN = (9, 7)  # inner corners (columns, rows) of the 10 x 8 square board
SQUARE_MM = 18.0  # measured printed square size, see docs/PLAN.md
IMAGE_GLOBS = ("calibration/images/calib_*.jpeg", "calibration/images/calib_*.jpg")
OUTPUT = "calibration/camera_params.npz"
REPORT = "calibration/calibration_report.txt"


def main():
    # Corner positions on the board in mm. The board is flat, so z = 0.
    board = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    board[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM

    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    object_points, image_points, used, skipped = [], [], [], []
    image_size = None

    for path in sorted(p for pattern in IMAGE_GLOBS for p in glob.glob(pattern)):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            skipped.append((path, "cannot be read"))
            continue
        size = gray.shape[::-1]  # (width, height)
        if image_size is None:
            image_size = size
        if size != image_size:
            skipped.append((path, f"size {size} differs from {image_size}"))
            continue
        found, corners = cv2.findChessboardCornersSB(gray, PATTERN, flags)
        if not found:
            skipped.append((path, "board not detected"))
            continue
        object_points.append(board)
        image_points.append(corners)
        used.append(path)
        print(f"detected: {path}")

    if len(used) < 10:
        sys.exit(f"Only {len(used)} usable images. Retake the board photos.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    # Reprojection error per image: how far the re-projected corners land
    # from the detected corners, in pixels.
    errors = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        errors.append(cv2.norm(img, projected, cv2.NORM_L2) / np.sqrt(len(obj)))

    np.savez(
        OUTPUT,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        rms_reprojection_error_px=rms,
        square_mm=SQUARE_MM,
    )

    lines = [
        f"images used: {len(used)}, skipped: {len(skipped)}",
        f"image size (w, h): {image_size}",
        f"square size: {SQUARE_MM} mm, pattern: {PATTERN} inner corners",
        f"overall RMS reprojection error: {rms:.4f} px",
        f"per-image error: mean {np.mean(errors):.4f} px, "
        f"median {np.median(errors):.4f} px, worst {np.max(errors):.4f} px",
        f"camera matrix:\n{camera_matrix}",
        f"distortion coefficients (k1 k2 p1 p2 k3):\n{dist_coeffs.ravel()}",
    ]
    lines += [f"skipped: {path} ({reason})" for path, reason in skipped]
    if len(used) < 20:
        lines.append("WARNING: fewer than 20 usable images")

    report = "\n".join(lines)
    print(report)
    with open(REPORT, "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()