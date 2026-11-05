"""Fine-tune Mask R-CNN on the box dataset.

The checkpoint is chosen on the val split only. The test split is evaluated once, at the end.
"""
import contextlib
import io
import json
import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms.functional import to_tensor

SEED = 42
EPOCHS = 20
BATCH_SIZE = 2
LEARNING_RATE = 0.005
IMAGE_DIR = "dataset/images"
LABELS = "dataset/labels/instances_default.json"
SPLIT_DIR = "dataset/splits"
OUT_DIR = "models/weights"
SCORE_THRESHOLD = 0.5


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mask_to_box(mask):
    ys, xs = torch.where(mask > 0)
    return torch.stack([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]).float()


class BoxDataset(torch.utils.data.Dataset):
    def __init__(self, coco, names, train):
        self.coco, self.train = coco, train
        by_name = {os.path.basename(i["file_name"]): i for i in coco.dataset["images"]}
        self.items = [by_name[n] for n in names]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        info = self.items[idx]
        path = os.path.join(IMAGE_DIR, os.path.basename(info["file_name"]))
        image = to_tensor(Image.open(path).convert("RGB"))
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=info["id"]))
        masks = torch.from_numpy(np.stack([self.coco.annToMask(a) for a in anns]))
        if self.train:
            if random.random() < 0.5:
                image, masks = image.flip(-1), masks.flip(-1)
            image = (image * random.uniform(0.7, 1.3)).clamp(0, 1)
        target = {
            "boxes": torch.stack([mask_to_box(m) for m in masks]),
            "labels": torch.ones(len(anns), dtype=torch.int64),
            "masks": masks,
        }
        return image, target


def build_model():
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 256, 2)
    return model


@torch.no_grad()
def predict(model, dataset, device):
    model.eval()
    results = []
    for i in range(len(dataset)):
        image, target = dataset[i]
        pred = model([image.to(device)])[0]
        results.append((dataset.items[i], target, {k: v.cpu() for k, v in pred.items()}))
    return results


def mask_iou(a, b):
    union = (a | b).sum().item()
    return (a & b).sum().item() / union if union else 0.0


def coco_ap(coco, results):
    cat_id = coco.getCatIds()[0]
    detections = []
    for info, _, pred in results:
        for score, mask in zip(pred["scores"], pred["masks"][:, 0] > 0.5):
            rle = mask_utils.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            detections.append({"image_id": info["id"], "category_id": cat_id,
                               "segmentation": rle, "score": float(score)})
    if not detections:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = COCOeval(coco, coco.loadRes(detections), "segm")
        evaluator.params.imgIds = [r[0]["id"] for r in results]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return float(evaluator.stats[1]), float(evaluator.stats[0])


def summarise(coco, results):
    tp = fp = fn = 0
    ious = []
    for _, target, pred in results:
        truth = target["masks"][0].bool()
        keep = pred["scores"] >= SCORE_THRESHOLD
        matched = False
        for mask in pred["masks"][keep, 0] > 0.5:
            if not matched and mask_iou(mask, truth) >= 0.5:
                tp += 1
                matched = True
            else:
                fp += 1
        fn += 0 if matched else 1
        top = pred["masks"][0, 0] > 0.5 if len(pred["scores"]) else None
        ious.append(mask_iou(top, truth) if top is not None else 0.0)
    ap50, ap = coco_ap(coco, results)
    return {
        "mask_iou_mean": float(np.mean(ious)),
        "mask_iou_median": float(np.median(ious)),
        "mask_iou_worst": float(np.min(ious)),
        "mAP@0.5": ap50,
        "mAP@0.5:0.95": ap,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "images": len(results),
    }


def read_split(name):
    with open(os.path.join(SPLIT_DIR, f"{name}.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    coco = COCO(LABELS)
    train_set = BoxDataset(coco, read_split("train"), train=True)
    val_set = BoxDataset(coco, read_split("val"), train=False)
    test_set = BoxDataset(coco, read_split("test"), train=False)
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    os.makedirs(OUT_DIR, exist_ok=True)
    best_path = os.path.join(OUT_DIR, "best.pt")
    best_val, history = -1.0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for images, targets in loader:
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        val = summarise(coco, predict(model, val_set, device))
        history.append({"epoch": epoch, "train_loss": total / len(loader), "val": val})
        print(f"epoch {epoch}: loss {total / len(loader):.4f}, val mAP@0.5:0.95 {val['mAP@0.5:0.95']:.4f}")
        if val["mAP@0.5:0.95"] > best_val:
            best_val = val["mAP@0.5:0.95"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test = summarise(coco, predict(model, test_set, device))
    print("TEST:", json.dumps(test, indent=2))

    log = {
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "best_val_mAP@0.5:0.95": best_val, "history": history, "test": test,
    }
    with open("models/train_log.json", "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()