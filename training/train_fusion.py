
import sys
sys.path.append('..')

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from app.core.config import settings
from app.models.embedding_extractor import get_embedding_extractor, cosine_similarity
from app.models.face_aligner import get_face_aligner
from app.models.face_detector import get_face_detector
from app.models.fusion_model import FusionMLP
from training.config_utils import load_training_config, set_global_seed
from training.experiment_tracker import ExperimentTracker

class EmbeddingPairDataset(Dataset):

    def __init__(self, embeddings_list, labels):
        self.embeddings_list = embeddings_list
        self.labels = labels

    def __len__(self):
        return len(self.embeddings_list)

    def __getitem__(self, idx):
        arcface, facenet, mobilefacenet = self.embeddings_list[idx]

        concat_emb = np.concatenate([arcface, facenet, mobilefacenet])

        return torch.from_numpy(concat_emb).float(), self.labels[idx]

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        normalized = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized, normalized.t())

        labels = labels.unsqueeze(1)
        same_identity = (labels == labels.t()).float()

        pos_loss = (1 - similarity_matrix) * same_identity

        neg_loss = F.relu(similarity_matrix - self.margin) * (1 - same_identity)

        loss = (pos_loss + neg_loss).mean()

        return loss

def extract_embeddings_from_images(image_dir: Path, detector, aligner, extractor):
    embeddings_list = []
    labels = []

    identity_dirs = [d for d in image_dir.iterdir() if d.is_dir()]

    print(f"Found {len(identity_dirs)} identities")

    for identity_idx, identity_dir in enumerate(tqdm(identity_dirs, desc="Processing identities")):
        image_files = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))

        for image_path in image_files:
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    continue

                detection = detector.detect_largest(image)
                if detection is None:
                    continue

                face_160 = aligner.align(image, detection, 160)
                face_224 = aligner.align(image, detection, 224)

                embeddings = extractor.extract_all_embeddings(face_160, face_224)

                if len(embeddings) == 3:
                    embeddings_list.append((
                        embeddings['arcface'],
                        embeddings['facenet'],
                        embeddings['mobilefacenet']
                    ))
                    labels.append(identity_idx)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

    print(f"Extracted embeddings for {len(embeddings_list)} images")

    return embeddings_list, labels

def generate_synthetic_data(num_identities=100, samples_per_identity=10):
    print(f"Generating synthetic data: {num_identities} identities, {samples_per_identity} samples each")

    embeddings_list = []
    labels = []

    for identity_idx in range(num_identities):
        arcface_proto = np.random.randn(512).astype(np.float32)
        facenet_proto = np.random.randn(512).astype(np.float32)
        mobile_proto = np.random.randn(512).astype(np.float32)

        arcface_proto = arcface_proto / (np.linalg.norm(arcface_proto) + 1e-8)
        facenet_proto = facenet_proto / (np.linalg.norm(facenet_proto) + 1e-8)
        mobile_proto = mobile_proto / (np.linalg.norm(mobile_proto) + 1e-8)

        for _ in range(samples_per_identity):
            noise_scale = 0.1

            arcface = arcface_proto + np.random.randn(512).astype(np.float32) * noise_scale
            facenet = facenet_proto + np.random.randn(512).astype(np.float32) * noise_scale
            mobile = mobile_proto + np.random.randn(512).astype(np.float32) * noise_scale

            arcface = arcface / (np.linalg.norm(arcface) + 1e-8)
            facenet = facenet / (np.linalg.norm(facenet) + 1e-8)
            mobile = mobile / (np.linalg.norm(mobile) + 1e-8)

            embeddings_list.append((arcface, facenet, mobile))
            labels.append(identity_idx)

    return embeddings_list, labels

def train_fusion_mlp(
    train_loader,
    val_loader,
    device,
    epochs=50,
    lr=0.001,
    save_path="weights/fusion_mlp.pth",
    tracker: ExperimentTracker | None = None,
) -> float:
    model = FusionMLP().to(device)

    criterion = ContrastiveLoss(margin=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')

    print(f"\nTraining on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for concat_emb, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            concat_emb = concat_emb.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            fused_embeddings = model(concat_emb)

            loss = criterion(fused_embeddings, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for concat_emb, labels in val_loader:
                concat_emb = concat_emb.to(device)
                labels = labels.to(device)

                fused_embeddings = model(concat_emb)
                loss = criterion(fused_embeddings, labels)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if tracker:
            tracker.log_epoch(
                epoch=epoch + 1,
                metrics={
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"✓ Model saved to {save_path}")

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    return float(best_val_loss)

def main():
    parser = argparse.ArgumentParser(description='Train Fusion MLP')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON/YAML config file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory with face images organized by identity')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data instead of real images')
    parser.add_argument('--num_identities', type=int, default=None,
                        help='Number of identities for synthetic data')
    parser.add_argument('--samples_per_id', type=int, default=None,
                        help='Samples per identity for synthetic data')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed override')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for trained model')

    args = parser.parse_args()

    config = load_training_config(args.config)
    fusion_cfg = config["train_fusion"]

    seed = args.seed if args.seed is not None else config["seed"]
    set_global_seed(seed)

    epochs = args.epochs or fusion_cfg["epochs"]
    batch_size = args.batch_size or fusion_cfg["batch_size"]
    lr = args.lr or fusion_cfg["learning_rate"]
    num_identities = args.num_identities or 200
    samples_per_id = args.samples_per_id or 20
    output_path = Path(args.output or Path(config["output_dir"]) / "fusion_mlp.pth")

    tracker = ExperimentTracker(
        experiment_name="fusion_mlp",
        output_dir=config["experiment_log_dir"],
        config={
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "data_dir": args.data_dir,
            "synthetic": args.synthetic or args.data_dir is None,
        },
    )

    device = settings.get_device()
    print(f"Using device: {device}")

    if args.synthetic or args.data_dir is None:
        print("\n=== Using Synthetic Data ===")
        embeddings_list, labels = generate_synthetic_data(
            num_identities=num_identities,
            samples_per_identity=samples_per_id
        )
    else:
        print(f"\n=== Extracting Embeddings from {args.data_dir} ===")
        detector = get_face_detector()
        aligner = get_face_aligner()
        extractor = get_embedding_extractor()

        embeddings_list, labels = extract_embeddings_from_images(
            Path(args.data_dir),
            detector,
            aligner,
            extractor
        )

    if not embeddings_list:
        raise RuntimeError("No embeddings extracted for training. Check input data or model configuration.")

    split_idx = int(len(embeddings_list) * 0.8)
    if split_idx == 0 or split_idx == len(embeddings_list):
        raise RuntimeError("Insufficient data to create train/validation splits. Collect more samples.")

    train_embeddings = embeddings_list[:split_idx]
    train_labels = labels[:split_idx]
    val_embeddings = embeddings_list[split_idx:]
    val_labels = labels[split_idx:]

    print(f"\nTrain samples: {len(train_embeddings)}")
    print(f"Val samples: {len(val_embeddings)}")

    train_dataset = EmbeddingPairDataset(train_embeddings, train_labels)
    val_dataset = EmbeddingPairDataset(val_embeddings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = train_fusion_mlp(
        train_loader,
        val_loader,
        device,
        epochs=epochs,
        lr=lr,
        save_path=str(output_path),
        tracker=tracker
    )

    tracker.finalize(
        {
            "best_val_loss": best_val_loss,
            "model_path": str(output_path),
        }
    )

if __name__ == "__main__":
    main()
