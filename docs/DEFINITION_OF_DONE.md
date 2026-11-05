# Definition of Done

Written before starting, ticked at the end with evidence. Unticked lines are not done.

- [x] Reprojection error reported, with calibration image count: 0.7605 px RMS, 24 of 25 images used (calibration/calibration_report.txt)
- [x] Test split committed before training: commit 1cea12c, files in dataset/splits/ (seed 42, 49 train / 14 val / 7 test)
- [x] Own dataset, 70 images, undistorted, labelled with masks: dataset/images_raw (not published, see README), dataset/labels/instances_default.json
- [x] Segmentation model trained and evaluated on the test split: models/train.py, models/train_log.json
- [x] Weights published with SHA-256: see README
- [ ] Command runs on a new image and prints width and height in mm: NOT DONE
- [ ] Accuracy table for 10 instances, raw outputs saved: NOT DONE
- [ ] Raw vs undistorted errors compared: NOT DONE
- [x] Everything I did not finish is listed: README, section "Not finished"
