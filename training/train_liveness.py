
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from app.models.liveness_detector import LivenessResNet
from app.utils.metrics import compute_eer, compute_far_frr, compute_roc_auc
from training.config_utils import load_training_config, set_global_seed
from training.dataset_loaders import get_liveness_dataloader
from training.experiment_tracker import ExperimentTracker

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': running_loss/total, 'acc': 100*correct/total})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
            live_scores = probabilities[:, 1].cpu().numpy()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(live_scores)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    eer, threshold = compute_eer(all_labels, all_scores)
    far, frr = compute_far_frr(all_labels, all_scores, threshold)
    auc = compute_roc_auc(all_labels, all_scores)

    return epoch_loss, epoch_acc, eer, far, frr, auc

def main():
    parser = argparse.ArgumentParser(description='Train Liveness Detector')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON/YAML config file')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed override')

    args = parser.parse_args()

    config = load_training_config(args.config)
    liveness_cfg = config["train_liveness"]

    seed = args.seed if args.seed is not None else config["seed"]
    set_global_seed(seed)

    device_choice = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_choice if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir or Path(config["output_dir"]) / "liveness")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training on device: {device}")
    print(f"Dataset: {args.data_root}")
    print(f"Output: {output_dir}")

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    batch_size = args.batch_size or liveness_cfg["batch_size"]
    num_workers = liveness_cfg["num_workers"]
    epochs = args.epochs or liveness_cfg["epochs"]
    lr = args.lr or liveness_cfg["learning_rate"]

    tracker = ExperimentTracker(
        experiment_name="liveness_resnet",
        output_dir=config["experiment_log_dir"],
        config={
            "seed": seed,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "data_root": args.data_root,
        },
    )

    train_loader = get_liveness_dataloader(
        args.data_root, 'train', batch_size, True, num_workers, train_transform
    )
    val_loader = get_liveness_dataloader(
        args.data_root, 'val', batch_size, False, num_workers, val_transform
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = LivenessResNet()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_eer = 1.0
    best_metrics = {}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 60)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, eer, far, frr, auc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"EER: {eer:.4f} | FAR: {far:.4f} | FRR: {frr:.4f} | AUC: {auc:.4f}")

        tracker.log_epoch(
            epoch=epoch + 1,
            metrics={
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "eer": float(eer),
                "far": float(far),
                "frr": float(frr),
                "auc": float(auc),
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
        )

        if eer < best_eer:
            best_eer = eer
            checkpoint_path = output_dir / 'liveness_resnet18_best.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved best model (EER: {eer:.4f})")
            best_metrics = {
                "checkpoint": str(checkpoint_path),
                "eer": float(eer),
                "far": float(far),
                "frr": float(frr),
                "auc": float(auc),
                "val_loss": float(val_loss),
            }

    print(f"\nTraining complete! Best EER: {best_eer:.4f}")
    tracker.finalize(
        {
            "best_eer": float(best_eer),
            **best_metrics,
        }
    )

if __name__ == '__main__':
    main()
