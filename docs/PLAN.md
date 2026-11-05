# Plan

## Setup
- Object: Samsung Galaxy A06 phone box, 172 x 82 x 28 mm (measured by me with a ruler, about +-0.5 mm). Rigid, rectangular, matte.
- Reference: student ID card (CNIC size), 81.28 x 50.8 mm (measured by me with a ruler, about +-0.5 mm), lying on the same box face as the object.
- Calibration board: printed checkerboard, 10 x 8 squares, 9 x 7 inner corners. Designed at 20 mm, measured 18 mm per square (printer scaled it), taped flat on cardboard.
- Camera: Samsung Galaxy A06 phone (4/64 GB). Focus, zoom and resolution locked for all photos.
- Laptop: HP, Intel Core i5 6th gen, 8 GB RAM, 256 GB SSD, Windows, no NVIDIA GPU.

## Order and time (8 h of work)
1. Calibration, 20+ board photos, OpenCV (1 h)
2. Photograph 70+ images of the box with the card, undistort, label masks (2.5 h)
3. Train a segmentation model: Mask R-CNN, torchvision, pretrained backbone (1 h of my time). No GPU on my laptop, so training runs on free Google Colab and is declared in the README. Inference and measurements run on my laptop CPU.
4. One command: image to millimetres, with object-not-found and reference-not-found cases (1.5 h)
5. Ruler accuracy table, 10+ instances, plus raw vs undistorted comparison (1 h)
6. Documents, README, Loom (1 h)

## Skipping for now
Speed target, breaking condition, refusal thresholds and the decision record (items 7 to 10), unless time remains.

## Split
Fixed in dataset/splits/ with seed 42 before any photo was taken. The test split is never used for training, tuning or picking a checkpoint.

## Update at the end
The plan changed: the one-command pipeline and everything after it were not reached. Finding the reference card with hand-written edge rules failed on drawn test scenes. The planned fix (label the card as a second class and retrain) did not fit in the time. Items 4 to 10 are listed as not finished in the README.