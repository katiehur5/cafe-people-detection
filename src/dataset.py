import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json

class CafeDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transforms=None):
        """
        Args:
            image_dir (str): path to images (train/valid/test)
            annotation_file (str): path to COCO JSON file
            transforms (callable, optional): image transforms
        """
        self.image_dir = image_dir
        self.transforms = transforms
        with open(annotation_file, 'r') as f:
            coco = json.load(f)
        self.images = {img["id"]: img for img in coco["images"]}

        self.annotations = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        anns = self.annotations.get(image_id, [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id])
        }

        if self.transforms:
            image = self.transforms(image)
        return image, target