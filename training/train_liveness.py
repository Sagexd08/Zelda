"""
Liveness Detection Training Script
Train CNN-based liveness detector on CASIA-FASD or CelebA-Spoof dataset
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datetime import datetime

from training.dataset_loaders import get_liveness_dataloader
from app.models.liveness_detector import LivenessResNet
from app.utils.metrics import compute_far_frr, compute_eer, compute_roc_auc


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
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
    """Validate model"""
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
    
    # Compute FAR, FRR, EER
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    eer, threshold = compute_eer(all_labels, all_scores)
    far, frr = compute_far_frr(all_labels, all_scores, threshold)
    auc = compute_roc_auc(all_labels, all_scores)
    
    return epoch_loss, epoch_acc, eer, far, frr, auc


def main():
    parser = argparse.ArgumentParser(description='Train Liveness Detector')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training on device: {device}")
    print(f"Dataset: {args.data_root}")
    print(f"Output: {args.output_dir}")
    
    # Transforms
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
    
    # Dataloaders
    train_loader = get_liveness_dataloader(
        args.data_root, 'train', args.batch_size, True, 4, train_transform
    )
    val_loader = get_liveness_dataloader(
        args.data_root, 'val', args.batch_size, False, 4, val_transform
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Model
    model = LivenessResNet()
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Training loop
    best_eer = 1.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, eer, far, frr, auc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"EER: {eer:.4f} | FAR: {far:.4f} | FRR: {frr:.4f} | AUC: {auc:.4f}")
        
        # Save best model
        if eer < best_eer:
            best_eer = eer
            checkpoint_path = output_dir / 'liveness_resnet18_best.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved best model (EER: {eer:.4f})")
    
    print(f"\nTraining complete! Best EER: {best_eer:.4f}")


if __name__ == '__main__':
    main()

