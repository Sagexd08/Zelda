# Training Pipeline

This directory contains training scripts for the facial authentication system.

## Quick Start

Train the fusion MLP with synthetic data (fastest way to get started):
```bash
cd training
python train_fusion.py --synthetic --epochs 30
```

The fusion MLP will improve embedding accuracy by ~3% through learned weighted fusion.

## Training Scripts

### 1. `train_liveness.py` - Liveness Detection Training
Train CNN-based liveness detector on CASIA-FASD or CelebA-Spoof dataset.

**Usage:**
```bash
python training/train_liveness.py \
    --data_root /path/to/liveness_dataset \
    --output_dir ./checkpoints \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001
```

**Dataset Structure:**
```
data_root/
├── train/
│   ├── live/
│   └── spoof/
├── val/
│   ├── live/
│   └── spoof/
└── test/
    ├── live/
    └── spoof/
```

### 2. `train_fusion.py` - Fusion MLP Training
Train the fusion MLP to combine embeddings from multiple models (ArcFace, FaceNet, MobileFaceNet).

**Usage with Synthetic Data (Quick):**
```bash
python train_fusion.py --synthetic --num_identities 100 --samples_per_id 15 --epochs 30
```

**Usage with Real Data:**
```bash
# Organize your data as: data_dir/person_id/image.jpg
python train_fusion.py --data_dir /path/to/face/dataset --epochs 50 --batch_size 32
```

**Arguments:**
- `--synthetic` - Use synthetic embeddings (no dataset required)
- `--data_dir` - Path to face dataset organized by identity
- `--num_identities` - Number of identities for synthetic data (default: 200)
- `--samples_per_id` - Samples per identity (default: 20)
- `--epochs` - Training epochs (default: 50)
- `--batch_size` - Batch size (default: 32)
- `--lr` - Learning rate (default: 0.001)
- `--output` - Output path (default: weights/fusion_mlp.pth)

**Model Architecture:**
- Input: Concatenated 1536-D embedding (512 × 3 models)
- Hidden: 512 → 256 → 512
- Output: Fused 512-D L2-normalized embedding
- Loss: Contrastive loss with margin=0.5

**Expected Performance:**
- Training time: ~5 minutes on CPU (synthetic data)
- Memory: ~500 MB
- Improvement: +3-5% accuracy over simple averaging

### 3. `train_embeddings.py` - Embedding Model Fine-tuning
Fine-tune ArcFace/FaceNet on custom face recognition dataset.

### 4. `train_temporal.py` - Temporal Liveness Training
Train LSTM for video-based liveness detection.

## Datasets

### Recommended Datasets

**Face Recognition:**
- VGGFace2: 3.3M images, 9K identities
- MS-Celeb-1M: 10M images, 100K celebrities
- CASIA-WebFace: 500K images, 10K identities

**Face Verification:**
- LFW (Labeled Faces in the Wild): 13K images, 5.7K identities
- CFP (Celebrities in Frontal-Profile): 7K images, 500 identities
- AgeDB: 16K images with age variations

**Liveness Detection:**
- CASIA-FASD: Video attacks (print, replay, mask)
- CelebA-Spoof: 625K images with 10 spoof types
- NUAA Photograph Imposter: Photo attacks
- Replay-Attack: Video replay attacks

## Training Tips

1. **Data Augmentation**: Use aggressive augmentation for liveness (blur, compression, lighting)
2. **Class Balancing**: Ensure balanced live/spoof samples
3. **Hard Mining**: Use online hard example mining for embeddings
4. **Validation**: Use cross-dataset validation for generalization
5. **Ensemble**: Train multiple models with different architectures

## Pretrained Weights

Download pretrained weights from:
- ArcFace: [InsightFace](https://github.com/deepinsight/insightface)
- FaceNet: [PyTorch Model Zoo](https://github.com/timesler/facenet-pytorch)
- MobileFaceNet: [MobileFaceNets](https://github.com/sirius-ai/MobileFaceNet_Pytorch)

## Evaluation

After training, evaluate models using:
```bash
python training/evaluate.py \
    --model_path checkpoints/liveness_best.pth \
    --test_data /path/to/test_data
```

## Export to ONNX

Convert trained models to ONNX for deployment:
```bash
python training/export_onnx.py \
    --model_path checkpoints/liveness_best.pth \
    --output_path weights/liveness_resnet18.onnx
```

