# Model Weights Directory

This directory contains pretrained model weights for the facial authentication system.

## Required Models

### 1. Face Recognition Models

#### ArcFace (ResNet100) - **REQUIRED**
- **Size**: 249 MB
- **Purpose**: Primary face embedding extraction
- **Accuracy**: 99.8% on LFW
- **Download**:
  ```bash
  wget https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_resnet100.pth \
    -O weights/arcface_resnet100.pth
  ```
- **Alternative**: [Google Drive](https://drive.google.com/file/d/XXXXX/view)

#### FaceNet (Inception-ResNet-v1) - **REQUIRED**
- **Size**: 107 MB
- **Purpose**: Secondary face embedding extraction
- **Accuracy**: 99.6% on LFW
- **Note**: Automatically downloaded by `facenet-pytorch` on first use
- **Manual Download**:
  ```bash
  wget https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt \
    -O weights/facenet_inception_resnet_v1.pth
  ```

#### MobileFaceNet - **REQUIRED**
- **Size**: 4 MB
- **Purpose**: Lightweight mobile face embedding
- **Accuracy**: 99.2% on LFW
- **Download**:
  ```bash
  wget https://github.com/sirius-ai/MobileFaceNet_Pytorch/raw/master/model.pth \
    -O weights/mobilefacenet.pth
  ```

### 2. Liveness Detection Models

#### Liveness CNN (ResNet-18) - **REQUIRED**
- **Size**: 45 MB
- **Purpose**: Detect photo/video/mask spoofs
- **Dataset**: Trained on CASIA-FASD + CelebA-Spoof
- **Training**: See `training/train_liveness.py`
- **Pretrained** (if available):
  ```bash
  wget https://example.com/liveness_resnet18.pth \
    -O weights/liveness_resnet18.pth
  ```
- **Note**: You need to train this model on your own dataset or use provided checkpoint

#### Temporal Liveness LSTM - OPTIONAL
- **Size**: 8 MB
- **Purpose**: Video-based liveness detection
- **Training**: See `training/train_temporal.py`
- **File**: `weights/temporal_lstm.pth`

### 3. Depth Estimation - OPTIONAL

#### MiDaS Small
- **Size**: 28 MB
- **Purpose**: Monocular depth estimation for 3D validation
- **Note**: Automatically downloaded by `torch.hub` on first use
- **Manual Download**:
  ```bash
  wget https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small-70d6b9c8.pt \
    -O weights/midas_v21_small.pt
  ```

### 4. Fusion Model - OPTIONAL

#### Embedding Fusion MLP
- **Size**: < 1 MB
- **Purpose**: Learned weighted fusion of multiple embeddings
- **Training**: See `training/train_fusion.py`
- **File**: `weights/fusion_mlp.pth`
- **Note**: If not available, system uses simple averaging

## Directory Structure

After downloading all models:

```
weights/
├── README.md (this file)
├── arcface_resnet100.pth          # 249 MB - REQUIRED
├── facenet_inception_resnet_v1.pth # 107 MB - REQUIRED  
├── mobilefacenet.pth               # 4 MB - REQUIRED
├── liveness_resnet18.pth          # 45 MB - REQUIRED
├── temporal_lstm.pth              # 8 MB - Optional
├── midas_v21_small.pt             # 28 MB - Optional
└── fusion_mlp.pth                 # <1 MB - Optional
```

## Model Checksums (MD5)

Verify downloaded models:

```bash
# ArcFace
md5sum arcface_resnet100.pth
# Expected: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# FaceNet
md5sum facenet_inception_resnet_v1.pth
# Expected: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# MobileFaceNet
md5sum mobilefacenet.pth
# Expected: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## Quick Download Script

```bash
#!/bin/bash
# download_models.sh

echo "Downloading model weights..."

# ArcFace
echo "Downloading ArcFace..."
wget -q --show-progress https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_resnet100.pth \
  -O weights/arcface_resnet100.pth

# MobileFaceNet
echo "Downloading MobileFaceNet..."
wget -q --show-progress https://github.com/sirius-ai/MobileFaceNet_Pytorch/raw/master/model.pth \
  -O weights/mobilefacenet.pth

echo "✓ Basic models downloaded!"
echo "Note: FaceNet and MiDaS will auto-download on first use"
echo "Note: Liveness model needs to be trained (see training/)"
```

Make executable and run:
```bash
chmod +x download_models.sh
./download_models.sh
```

## Troubleshooting

### Issue: Models not found
**Solution**: Ensure `MODEL_BASE_PATH` in `.env` points to this directory:
```
MODEL_BASE_PATH=./weights
```

### Issue: Out of memory
**Solution**: Models are lazy-loaded. Only loaded models consume memory.

### Issue: Slow inference
**Solution**: 
1. Use GPU: Set `DEVICE=cuda` in `.env`
2. Export to ONNX for optimized inference
3. Use quantized models (INT8)

## Model Licenses

- **ArcFace**: Apache 2.0
- **FaceNet**: MIT
- **MobileFaceNet**: Apache 2.0
- **MiDaS**: MIT

Always check individual model licenses before commercial use.

## Custom Models

To use your own trained models:

1. Place model file in `weights/` directory
2. Update model path in `.env`:
   ```
   ARCFACE_MODEL=my_custom_arcface.pth
   ```
3. Ensure model architecture matches expected format

