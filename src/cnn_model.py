# src/cnn_model.py
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREPROCESSED_DIR = Path("preprocessed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_dataloaders(batch_size=16):
    # ImageNet normalization for pretrained models
    img_transforms = {
        "train": transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    }

    image_datasets = {}
    dataloaders = {}
    for split in ["train", "val", "test"]:
        split_dir = PREPROCESSED_DIR / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist")
        dataset = datasets.ImageFolder(split_dir, transform=img_transforms[split])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
        image_datasets[split] = dataset
        dataloaders[split] = loader

    return image_datasets, dataloaders


def build_model(num_classes=2, freeze_base=True):
    # ResNet18 with pretrained weights
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if freeze_base:
        for param in resnet.parameters():
            param.requires_grad = False

    # Replace final FC layer
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)

    return resnet.to(DEVICE)


def train_model(model, dataloaders, dataset_sizes, num_epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for phase in ["train", "val"]:
            if phase not in dataloaders:
                continue

            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            loop = tqdm(dataloaders[phase], desc=f"{phase}", leave=False)
            for inputs, labels in loop:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best val acc: {best_acc:.4f}")
    else:
        print("Warning: no best_state saved (maybe no val set?)")

    return model


def main():
    image_datasets, dataloaders = get_dataloaders(batch_size=16)
    dataset_sizes = {split: len(ds) for split, ds in image_datasets.items()}
    print("Dataset sizes:", dataset_sizes)

    model = build_model(num_classes=2, freeze_base=True)
    model = train_model(model, dataloaders, dataset_sizes, num_epochs=5, lr=1e-3)

    out_path = MODELS_DIR / "cnn_best.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Saved best CNN model to {out_path}")


if __name__ == "__main__":
    main()
