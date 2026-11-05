"""Undistort every raw dataset image with the saved camera calibration."""
import glob
import os

import cv2
import numpy as np

from calibration.undistort import load_params, undistort

RAW_DIR = "dataset/images_raw"
OUT_DIR = "dataset/images"


def main():
    camera_matrix, dist_coeffs = load_params()
    expected = tuple(int(v) for v in np.load("calibration/camera_params.npz")["image_size"])
    os.makedirs(OUT_DIR, exist_ok=True)

    paths = sorted(glob.glob(f"{RAW_DIR}/img_*.jpg"))
    for path in paths:
        image = cv2.imread(path)
        if image.shape[1::-1] != expected:
            raise SystemExit(f"{path} is {image.shape[1::-1]}, calibration needs {expected}")
        fixed = undistort(image, camera_matrix, dist_coeffs)
        cv2.imwrite(os.path.join(OUT_DIR, os.path.basename(path)), fixed,
                    [cv2.IMWRITE_JPEG_QUALITY, 98])
    print(f"undistorted {len(paths)} images into {OUT_DIR}")


if __name__ == "__main__":
    main()