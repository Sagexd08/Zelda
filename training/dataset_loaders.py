
from typing import Tuple, List, Optional
from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch

class FaceVerificationDataset(Dataset):

    def __init__(
        self,
        data_root: str,
        pairs_file: str,
        transform=None
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.pairs = self._load_pairs(pairs_file)

    def _load_pairs(self, pairs_file: str) -> List[Tuple]:
        pairs = []

        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    name1, img1, name2, img2, label = parts[:5]
                    pairs.append((name1, img1, name2, img2, int(label)))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple:
        name1, img1, name2, img2, label = self.pairs[idx]

        img1_path = self.data_root / name1 / img1
        img2_path = self.data_root / name2 / img2

        image1 = cv2.imread(str(img1_path))
        image2 = cv2.imread(str(img2_path))

        if image1 is None or image2 is None:
            return torch.zeros(3, 160, 160), torch.zeros(3, 160, 160), label

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label

class LivenessDataset(Dataset):

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform=None
    ):
        self.data_root = Path(data_root) / split
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []

        live_dir = self.data_root / 'live'
        if live_dir.exists():
            for img_path in live_dir.glob('**/*.jpg'):
                samples.append((str(img_path), 1))
            for img_path in live_dir.glob('**/*.png'):
                samples.append((str(img_path), 1))

        spoof_dir = self.data_root / 'spoof'
        if spoof_dir.exists():
            for img_path in spoof_dir.glob('**/*.jpg'):
                samples.append((str(img_path), 0))
            for img_path in spoof_dir.glob('**/*.png'):
                samples.append((str(img_path), 0))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]

        image = cv2.imread(img_path)

        if image is None:
            return torch.zeros(3, 224, 224), label

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label

class FaceRecognitionDataset(Dataset):

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform=None
    ):
        self.data_root = Path(data_root) / split
        self.transform = transform
        self.samples, self.class_to_idx = self._load_samples()
        self.num_classes = len(self.class_to_idx)

    def _load_samples(self) -> Tuple[List, dict]:
        samples = []
        class_to_idx = {}

        for idx, identity_dir in enumerate(sorted(self.data_root.iterdir())):
            if not identity_dir.is_dir():
                continue

            class_to_idx[identity_dir.name] = idx

            for img_path in identity_dir.glob('*.jpg'):
                samples.append((str(img_path), idx))
            for img_path in identity_dir.glob('*.png'):
                samples.append((str(img_path), idx))

        return samples, class_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]

        image = cv2.imread(img_path)

        if image is None:
            return torch.zeros(3, 160, 160), label

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label

class TemporalLivenessDataset(Dataset):

    def __init__(
        self,
        data_root: str,
        sequence_length: int = 30,
        split: str = 'train'
    ):
        self.data_root = Path(data_root) / split
        self.sequence_length = sequence_length
        self.videos = self._load_videos()

    def _load_videos(self) -> List[Tuple[str, int]]:
        videos = []

        live_dir = self.data_root / 'live'
        if live_dir.exists():
            for video_path in live_dir.glob('**/*.mp4'):
                videos.append((str(video_path), 1))
            for video_path in live_dir.glob('**/*.avi'):
                videos.append((str(video_path), 1))

        spoof_dir = self.data_root / 'spoof'
        if spoof_dir.exists():
            for video_path in spoof_dir.glob('**/*.mp4'):
                videos.append((str(video_path), 0))
            for video_path in spoof_dir.glob('**/*.avi'):
                videos.append((str(video_path), 0))

        return videos

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> Tuple:
        video_path, label = self.videos[idx]

        frames = self._load_video_frames(video_path)

        if len(frames) == 0:
            return torch.zeros(self.sequence_length, 20), label

        features = np.random.randn(min(len(frames), self.sequence_length), 20).astype(np.float32)

        if len(features) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(features), 20), dtype=np.float32)
            features = np.vstack([features, padding])

        return torch.from_numpy(features), label

    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        frames = []

        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        return frames

def get_verification_dataloader(
    data_root: str,
    pairs_file: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None
) -> DataLoader:
    dataset = FaceVerificationDataset(data_root, pairs_file, transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def get_liveness_dataloader(
    data_root: str,
    split: str = 'train',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None
) -> DataLoader:
    dataset = LivenessDataset(data_root, split, transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
