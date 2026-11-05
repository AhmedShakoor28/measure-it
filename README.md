# Measure It: segmentation and metric measurement

Goal: report the real width and height of an object in millimetres from one photo. This submission is **partly finished**: calibration, dataset and segmentation model are done; the image-to-millimetres command and the accuracy checks are not. See "Not finished".

## What is here

| Folder | Content |
|---|---|
| calibration/ | 25 checkerboard photos, calibrate.py, undistort.py, saved camera_params.npz, calibration_report.txt |
| dataset/ | prepare_images.py, labels (COCO), split lists in splits/ |
| models/ | train.py and train_log.json (training config, curves, test metrics) |
| docs/ | PLAN.md, OBJECTIVES.md, DEFINITION_OF_DONE.md |

## Hardware

- Laptop: HP, Intel Core i5 6th gen, 8 GB RAM, 256 GB SSD, Windows, no GPU. Calibration and undistortion ran here.
- Training ran on Google Colab, NVIDIA T4. The training log records the torch and torchvision versions.
- Camera: Samsung Galaxy A06 (4/64 GB), rear camera, 1280 x 576 px, focus, zoom and resolution locked.

## Setup and how to run

Python 3.10.10 on Windows.

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python calibration\calibrate.py          # camera parameters and reprojection error
python -m dataset.prepare_images         # undistort dataset/images_raw into dataset/images
```

Training (Colab, GPU runtime, after unzipping the undistorted images into dataset/images):

```
pip install pycocotools
python models/train.py
```

Seed 42 is fixed, but GPU training may not reproduce the digits exactly.

## Dataset

- Object: Samsung Galaxy A06 phone box, with a student ID card (81.28 x 50.8 mm) lying on its face as the reference.
- 70 images I took, 1280 x 576, undistorted with the calibration. Split 49 train / 14 val / 7 test, fixed before any photo was taken (dataset/splits/, commit 1cea12c).
- Labelled with CVAT as one polygon per image around the box face, class `box`, exported as COCO 1.0. Four polygons have 5 or 6 points, the rest 4.
- The photos are not in the repository because they show personal details on the ID card. They are available on request. An earlier commit that contained them was removed from the branch.

## Calibration

24 of 25 checkerboard photos used, 18.0 mm squares measured with a ruler, 9 x 7 inner corners. Overall RMS reprojection error 0.7605 px (per image: mean 0.72, median 0.67, worst 1.22). Raw and undistorted images look the same to the eye, so this lens has little distortion. I did not measure how much undistortion changes the measurement error.

## Model

Mask R-CNN ResNet50-FPN (torchvision), fine-tuned from COCO weights on the 49 train images, 20 epochs, checkpoint chosen on the 14 val images only. I chose it because it gives masks and the mAP numbers the task asks for directly, and a pretrained backbone works with 49 images. Test results on 7 images are in docs/OBJECTIVES.md and models/train_log.json.

Weights: (https://github.com/AhmedShakoor28/measure-it/releases/tag/v1.0)

SHA-256 of best.pt: 4EFEEF27316A00F7C2A1FC5D5AA96FE912A8C527413D62197DFE12153A36F512 

## Not finished

- The one-command image-to-millimetres pipeline (items 4 and the "object not found / reference not found" cases).
- Accuracy against a ruler or calliper (item 5). No error number is claimed.
- Raw vs undistorted comparison (item 6), a speed or accuracy target (7), the condition that breaks it (8), refusing untrusted measurements (9) and the decision record (10).

What I tried: finding the reference card inside the photo with hand-written edge rules. On drawn test scenes it failed because the printed pictures on the box touch the card outline. The planned fix was to label the card as a second class and retrain; I ran out of time before doing it, and that code is not in this repository.
