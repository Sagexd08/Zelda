"""
Train temporal LSTM for video-based liveness detection
"""
import sys
sys.path.append('..')

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from app.core.config import settings
from app.models.temporal_liveness import TemporalLSTM
from training.config_utils import load_training_config, set_global_seed
from training.dataset_loaders import get_temporal_liveness_dataloader
from training.experiment_tracker import ExperimentTracker


def train_temporal_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.001,
    save_path: str = "weights/temporal_lstm.pth",
    tracker: ExperimentTracker | None = None,
):
    """
    Train temporal LSTM model
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device for training
        epochs: Number of training epochs
        lr: Learning rate
        save_path: Path to save trained model
    """
    # Initialize model
    model = TemporalLSTM(input_dim=512, hidden_dim=128, num_layers=2, num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    
    print(f"\nTraining on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_metrics = {}
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            frames = frames.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)
                
                outputs = model(frames)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Accuracy = {accuracy:.2f}%")

        if tracker:
            tracker.log_epoch(
                epoch=epoch + 1,
                metrics={
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "accuracy": float(accuracy),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
            )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"✓ Model saved to {save_path}")
            best_metrics = {
                "val_loss": float(val_loss),
                "accuracy": float(accuracy),
            }
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    if tracker:
        tracker.finalize(
            {
                "best_val_loss": float(best_val_loss),
                "model_path": save_path,
                **best_metrics,
            }
        )
    return float(best_val_loss)


def main():
    parser = argparse.ArgumentParser(description='Train Temporal LSTM')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to JSON/YAML config file')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory with video data')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=None,
                       help='Video sequence length')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for trained model')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed override')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of dataloader workers')
    
    args = parser.parse_args()

    config = load_training_config(args.config)
    temporal_cfg = config["train_temporal"]

    seed = args.seed if args.seed is not None else config["seed"]
    set_global_seed(seed)

    epochs = args.epochs or temporal_cfg["epochs"]
    batch_size = args.batch_size or temporal_cfg["batch_size"]
    lr = args.lr or temporal_cfg["learning_rate"]
    sequence_length = args.sequence_length or  temporal_cfg.get("sequence_length", 30)
    num_workers = args.num_workers or temporal_cfg["num_workers"]
    save_path = args.output or str(Path(config["output_dir"]) / "temporal_lstm.pth")

    device = settings.get_device()
    print(f"Using device: {device}")

    tracker = ExperimentTracker(
        experiment_name="temporal_lstm",
        output_dir=config["experiment_log_dir"],
        config={
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "sequence_length": sequence_length,
            "data_root": args.data_root,
        },
    )

    train_loader = get_temporal_liveness_dataloader(
        args.data_root,
        split='train',
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        sequence_length=sequence_length
    )
    val_loader = get_temporal_liveness_dataloader(
        args.data_root,
        split='val',
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sequence_length=sequence_length
    )

    train_temporal_lstm(
        train_loader,
        val_loader,
        device,
        epochs=epochs,
        lr=lr,
        save_path=save_path,
        tracker=tracker
    )


if __name__ == "__main__":
    main()

