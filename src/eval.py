# qualitative evaluation: runs inference, draws bboxes, counts people, saves images for visual evaluation

import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import ImageDraw
import numpy as np


from dataset import CafeDataset
from model import get_model

def collate_fn(batch):
    return tuple(zip(*batch))

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--score_thresh", type=float, default=0.5, help="Score threshold for detections")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_img_dir = "data/images/test"
    test_ann = "data/annotations/test.json"

    transform = T.Compose([T.ToTensor()])

    test_dataset = CafeDataset(test_img_dir, test_ann, transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs("outputs/figures", exist_ok=True)

    total_preds = 0
    total_kept = 0

    true_counts = []
    pred_counts = []

    for i, (images, targets) in enumerate(test_loader):
        image = images[0].to(device)
        target = targets[0]

        #inference
        outputs = model([image])[0]
        boxes = outputs["boxes"].detach().cpu()
        scores = outputs["scores"].detach().cpu()

        # score thresholding
        total_preds += len(scores)
        keep = scores >= args.score_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        total_kept += len(scores)

        pil_img = T.ToPILImage()(image.cpu())
        draw = ImageDraw.Draw(pil_img)
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], width=2)
            text = f"person {score:.2f}"
            draw.text((x1, max(0, y1 - 12)), text)
        
        pred_count = len(scores)

        # ground-truth count = number of GT boxes
        true_count = int(target["boxes"].shape[0])
        true_counts.append(true_count)
        pred_counts.append(pred_count)

        # save visualization
        img_id = int(target["image_id"].item()) if "image_id" in target else i
        out_path = f"outputs/figures/test_img_{img_id}_count_{pred_count}.png"
        pil_img.save(out_path)

        print(f"[{i+1}/{len(test_loader)}] image_id={img_id} predicted_people={pred_count} saved={out_path}")

    print("\n=== Summary ===")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Score threshold: {args.score_thresh}")
    print(f"Raw detections (before threshold): {total_preds}")
    print(f"Detections kept (after threshold): {total_kept}")
    print(f"Avg kept per image: {total_kept / max(1, len(test_dataset)):.2f}")

    true_counts = np.array(true_counts)
    pred_counts = np.array(pred_counts)

    mae = float(np.mean(np.abs(pred_counts - true_counts)))
    rmse = float(np.sqrt(np.mean((pred_counts - true_counts) ** 2)))

    print("\n=== Counting Metrics ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()