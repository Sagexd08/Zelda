# Training Pipeline

This directory contains training scripts for the facial authentication system.

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

### 2. `train_embeddings.py` - Embedding Model Fine-tuning
Fine-tune ArcFace/FaceNet on custom face recognition dataset.

### 3. `train_fusion.py` - Fusion Model Training
Train MLP fusion model on verification pairs.

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

