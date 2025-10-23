# Data Directory

This directory contains datasets for training and evaluation.

## Directory Structure

```
data/
├── README.md (this file)
├── raw/                 # Raw downloaded datasets
├── processed/           # Preprocessed data
├── train/              # Training split
├── val/                # Validation split
├── test/               # Test split
└── custom/             # Custom user data
```

## Recommended Datasets

### 1. Face Recognition

#### LFW (Labeled Faces in the Wild)
- **Size**: 1.3 GB (13,233 images)
- **Identities**: 5,749
- **Purpose**: Face verification evaluation
- **Download**: http://vis-www.cs.umass.edu/lfw/
- **Structure**:
  ```
  data/lfw/
  ├── pairs.txt
  └── lfw/
      ├── person1/
      │   ├── person1_0001.jpg
      │   └── person1_0002.jpg
      └── person2/
          └── ...
  ```

#### VGGFace2
- **Size**: 36 GB (3.3M images)
- **Identities**: 9,131
- **Purpose**: Face recognition training
- **Download**: https://github.com/ox-vgg/vgg_face2
- **Note**: Requires registration

#### MS-Celeb-1M
- **Size**: 100 GB (10M images)
- **Identities**: 100K
- **Purpose**: Large-scale face recognition
- **Note**: Dataset removed; use alternatives like VGGFace2

### 2. Liveness Detection

#### CASIA-FASD (Face Anti-Spoofing Database)
- **Size**: 2.5 GB
- **Videos**: 600 (50 subjects)
- **Attack Types**: Photo print, video replay, mask
- **Purpose**: Liveness detection training
- **Download**: http://www.cbsr.ia.ac.cn/english/FASDB_Agreement/Agreement.pdf
- **Structure**:
  ```
  data/casia_fasd/
  ├── train/
  │   ├── live/
  │   │   ├── video001.mp4
  │   │   └── ...
  │   └── spoof/
  │       ├── print_attack001.mp4
  │       └── ...
  ├── val/
  └── test/
  ```

#### CelebA-Spoof
- **Size**: 10 GB (625K images)
- **Subjects**: 10,177
- **Spoof Types**: 10 (print, replay, paper mask, etc.)
- **Purpose**: Advanced liveness training
- **Download**: https://github.com/Davidzhangyuanhan/CelebA-Spoof
- **Structure**:
  ```
  data/celeba_spoof/
  ├── Data/
  │   ├── train/
  │   │   ├── live/
  │   │   └── spoof/
  │   └── test/
  └── metas/
      └── intra_test.txt
  ```

### 3. Face Verification

#### CFP (Celebrities in Frontal-Profile)
- **Size**: 500 MB (7,000 images)
- **Identities**: 500
- **Purpose**: Profile face verification
- **Download**: http://www.cfpw.io/

#### AgeDB
- **Size**: 800 MB (16,488 images)
- **Identities**: 568
- **Purpose**: Age-invariant verification
- **Download**: https://ibug.doc.ic.ac.uk/resources/agedb/

## Data Preprocessing

### Face Alignment

```python
from app.models.face_detector import get_face_detector
from app.models.face_aligner import get_face_aligner
import cv2

detector = get_face_detector()
aligner = get_face_aligner()

# Process image
image = cv2.imread('data/raw/sample.jpg')
detection = detector.detect_largest(image)

if detection:
    aligned = aligner.align(image, detection, output_size=160)
    cv2.imwrite('data/processed/aligned.jpg', aligned)
```

### Batch Processing

```bash
# Process entire directory
python scripts/preprocess_dataset.py \
  --input_dir data/raw/lfw \
  --output_dir data/processed/lfw_aligned \
  --face_size 160
```

## Custom Dataset Format

For training your own models:

### Face Recognition Format
```
data/custom/recognition/
├── train/
│   ├── person1/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── person2/
│   └── ...
└── test/
    └── (same structure)
```

### Verification Pairs Format
Create `pairs.txt`:
```
# Same person
person1 img001.jpg person1 img002.jpg 1
person2 img001.jpg person2 img003.jpg 1

# Different persons
person1 img001.jpg person3 img001.jpg 0
person2 img002.jpg person4 img002.jpg 0
```

### Liveness Detection Format
```
data/custom/liveness/
├── train/
│   ├── live/
│   │   ├── sample001.jpg
│   │   └── ...
│   └── spoof/
│       ├── attack001.jpg
│       └── ...
├── val/
└── test/
```

## Data Augmentation

### Training Augmentation
- Random horizontal flip
- Random brightness (±20%)
- Random contrast (±20%)
- Gaussian blur (20% probability)
- JPEG compression artifacts
- Random occlusion patches

### Testing Augmentation
- Resize and normalize only
- No random augmentation

## Dataset Statistics

| Dataset | Images | Identities | Type | Use Case |
|---------|--------|------------|------|----------|
| LFW | 13K | 5.7K | Verification | Evaluation |
| VGGFace2 | 3.3M | 9K | Recognition | Training |
| CASIA-FASD | 600 videos | 50 | Liveness | Training |
| CelebA-Spoof | 625K | 10K | Liveness | Training |
| CFP | 7K | 500 | Verification | Evaluation |

## Download Scripts

### LFW Dataset
```bash
#!/bin/bash
# download_lfw.sh

mkdir -p data/lfw
cd data/lfw

# Download images
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xzf lfw.tgz

# Download pairs
wget http://vis-www.cs.umass.edu/lfw/pairs.txt

echo "✓ LFW dataset downloaded"
```

### CASIA-FASD Dataset
```bash
# Requires registration and agreement
# Visit: http://www.cbsr.ia.ac.cn/english/FASDB_Agreement/Agreement.pdf
```

## Data Privacy & Ethics

### Guidelines
1. **Consent**: Ensure proper consent for face data collection
2. **Privacy**: Anonymize data when possible
3. **Bias**: Include diverse demographics
4. **Retention**: Define clear data retention policies
5. **Compliance**: Follow GDPR, CCPA, BIPA regulations

### Face Data Best Practices
- ✅ Obtain explicit consent
- ✅ Provide data deletion mechanisms
- ✅ Encrypt stored images
- ✅ Audit data access
- ✅ Regular bias testing
- ❌ No collection without consent
- ❌ No sharing without permission
- ❌ No permanent storage of raw images

## Troubleshooting

### Issue: Dataset too large
**Solution**: Use subset or compressed version

### Issue: Out of disk space
**Solution**: Process and delete raw data incrementally

### Issue: Slow data loading
**Solution**: 
1. Use SSD storage
2. Enable multi-worker data loading
3. Preprocess and cache aligned faces

## Citation

If using these datasets in research, please cite:

**LFW**:
```bibtex
@article{huang2008labeled,
  title={Labeled faces in the wild: A database for studying face recognition in unconstrained environments},
  author={Huang, Gary B and others},
  year={2008}
}
```

**CASIA-FASD**:
```bibtex
@inproceedings{zhang2012face,
  title={A face antispoofing database with diverse attacks},
  author={Zhang, Zhiwei and others},
  year={2012}
}
```

