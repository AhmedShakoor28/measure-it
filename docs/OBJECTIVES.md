# Objectives

MO-1 was written before any accuracy measurement. The MO-2 target was written after I had seen the results of the one training run, so it is not an independent check.

## Setup behind every number

- Laptop (calibration, undistortion): HP, Intel Core i5 6th gen, 8 GB RAM, Windows, no GPU.
- Training only: Google Colab free tier, NVIDIA T4 GPU.
- Camera: Samsung Galaxy A06, rear camera, 1280 x 576 px photos, focus, zoom and resolution locked.
- Object: Samsung Galaxy A06 phone box, 172 x 82 mm, measured with a ruler (about +-0.5 mm).
- Reference: student ID card, 81.28 x 50.8 mm, measured with a ruler (about +-0.5 mm), lying on the box face.

## MO-1: measurement accuracy

| Field | Entry |
|---|---|
| What is measured | Absolute error, in mm, of the width and height the system reports for the box, against my ruler measurement of the same box. |
| How | A script runs the pipeline on each photo and compares with hand-measured values in a CSV. Every instance is listed, then mean, median and worst error. |
| Target | Mean absolute error of 2.0 mm or less for width and for height, no single error above 5 mm, over 10 or more instances. |
| Conditions | Phone about 40 cm straight above the box, card lying on the box face, dark matte background, indoor light, 1280 x 576 photos from the calibrated camera. |
| Not included | Photos where the box or card is partly out of frame or hidden, tilted camera, and photos the system refuses to measure (counted, not dropped). |

Why 2 mm: my ruler is good to about 0.5 mm and one pixel is about 0.33 mm, so about 1 mm is the floor of what I can claim. 2 mm is about 1.2% of the 172 mm side.

**Result: not measured.** The image-to-millimetres command was not finished, so there is no accuracy table. No accuracy claim is made.

## MO-2: segmentation quality

| Field | Entry |
|---|---|
| What is measured | Mean mask IoU of the box on the 7 images of the test split. |
| How | models/train.py picks the checkpoint on the val split only, then evaluates once on dataset/splits/test.txt against my hand-drawn masks. |
| Target | 0.90 or more. |
| Conditions | Same photos and camera as above, one training run, seed 42, Mask R-CNN ResNet50-FPN fine-tuned from COCO weights. |
| Not included | The 63 train and val images, and detections with a score below 0.5. |

**Result: target met on 7 images.** Raw output is in models/train_log.json.

| Test metric (7 images) | Value |
|---|---|
| Mask IoU mean / median / worst | 0.957 / 0.957 / 0.932 |
| mAP@0.5 | 1.0 |
| mAP@0.5:0.95 | 0.971 |
| Precision / recall | 1.0 / 1.0 |

Seven test images are few and the scene is simple (one rigid box on a plain dark background), so this shows the model works on this scene, not that it works in general.
