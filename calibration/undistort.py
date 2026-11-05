"""Undistort images with the saved camera calibration."""
import sys

import cv2
import numpy as np

PARAMS = "calibration/camera_params.npz"


def load_params(path=PARAMS):
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"]


def undistort(image, camera_matrix, dist_coeffs):
    return cv2.undistort(image, camera_matrix, dist_coeffs)


if __name__ == "__main__":
    source, target = sys.argv[1], sys.argv[2]
    camera_matrix, dist_coeffs = load_params()
    cv2.imwrite(target, undistort(cv2.imread(source), camera_matrix, dist_coeffs))