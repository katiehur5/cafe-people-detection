import torch
from torch.utils.data import DataLoader
import os
import torchvision.transforms as T

from dataset import CafeDataset
from model import get_model

# hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LOG_EVERY = 10  # batches

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device: {device}")

    train_img_dir = "data/images/train"
    train_ann = "data/annotations/train.json"

    val_img_dir = "data/images/valid"
    val_ann = "data/annotations/valid.json"

    transform = T.Compose([T.ToTensor()])

    print("[data] loading datasets...")
    train_dataset = CafeDataset(train_img_dir, train_ann, transforms=transform)
    val_dataset = CafeDataset(val_img_dir, val_ann, transforms=transform)
    print(f"[data] train samples: {len(train_dataset)}, val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    print(f"[data] train batches per epoch: {len(train_loader)}")

    model = get_model(num_classes=2)
    model.to(device)
    print("[model] model initialized and moved to device")

    #optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        LEARNING_RATE,
        MOMENTUM,
        WEIGHT_DECAY
    )

    os.makedirs("outputs/checkpoints", exist_ok=True)
    print("[train] starting training loop")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        batch_idx = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            batch_idx += 1

            if batch_idx % LOG_EVERY == 0:
                print(f"[train] epoch {epoch+1}/{NUM_EPOCHS}, batch {batch_idx}/{len(train_loader)}, loss={losses.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Avg loss: {avg_loss:.4f}")

        torch.save(
            model.state_dict(),
            f"outputs/checkpoints/model_epoch_{epoch+1}.pth"
        )
        print(f"[checkpoint] saved epoch {epoch+1} to outputs/checkpoints/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()