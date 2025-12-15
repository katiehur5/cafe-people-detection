# quantitative evaluation: compare predictions to ground truth, compute mAP@0.5 and other metrics

import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset import CafeDataset
from model import get_model

def collate_fn(batch):
    return tuple(zip(*batch))

# x1, y1, x2, y2 --> x, y, w, h
def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model .pth checkpoint")
    parser.add_argument("--imgdir", type=str, default="data/images/test", help="Test images directory")
    parser.add_argument("--ann", type=str, default="data/annotations/test.json", help="COCO test annotation JSON")
    parser.add_argument("--score_thresh", type=float, default=0.05, help="Score threshold for keeping detections")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loads ground truth
    coco_gt = COCO(args.ann)

    #debugging
    # print("GT num images:", len(coco_gt.getImgIds()))
    # print("GT num anns:", len(coco_gt.getAnnIds()))
    # print("GT catIds:", coco_gt.getCatIds())
    # print("First 10 imageIds:", coco_gt.getImgIds()[:10])

    cat_id = 1

    transform = T.Compose([T.ToTensor()])
    test_dataset = CafeDataset(args.imgdir, args.ann, transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)
    model.eval()

    detections = []

    for images, targets in test_loader:

        image = images[0].to(device)
        target = targets[0]
        
        image_id = int(target["image_id"].item())
        # debugging
        # print("pred image_id example:", image_id)
        # break

        out = model([image])[0]
        boxes = out["boxes"].detach().cpu()
        scores = out["scores"].detach().cpu()

        keep = scores >= args.score_thresh
        boxes = boxes[keep]
        scores = scores[keep]

        for box, score in zip(boxes, scores):
            det = {
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": xyxy_to_xywh(box.tolist()),
                "score": float(score.item()),
            }
            detections.append(det)

    if len(detections) == 0:
        print("No detections kept. Try lowering --score_thresh (e.g., 0.01).")
        return

    coco_dt = coco_gt.loadRes(detections)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

    #evaluate at IoU = .5 to get mAP@0.5
    coco_eval.params.iouThrs = [.5]
    coco_eval.params.catIds = [cat_id]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap50 = coco_eval.stats[0]
    print(f"\nAP@0.50 (mAP@0.5 for single class): {ap50:.4f}")

if __name__ == "__main__":
    main()