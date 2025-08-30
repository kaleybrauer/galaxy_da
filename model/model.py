import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def give_labels():
    CLASSES = ["elliptical", "irregular", "spiral"] 
    label2idx = {c:i for i,c in enumerate(CLASSES)}
    idx2label = {i:c for c,i in label2idx.items()}
    
    return label2idx, idx2label

# ======For source Dataset class ======
class GalaxyDataset_source(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, include_rotations=True):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.rotations = [0, 90, 180, 270]
        self.include_rotations = include_rotations

        # Map string labels to integers
        self.label2idx, self.idx2label = give_labels()
        self.df["label"] = self.df["classification"].str.strip().str.lower().map(self.label2idx)

    def __len__(self):
        if self.include_rotations:
          length = len(self.df) * 8  # 4 rotations * 2 flips
        else:
          length = len(self.df)

        return length

    def __getitem__(self, idx):
        if self.include_rotations:
          base_idx = idx // 8
          aug_idx = idx % 8

          row = self.df.iloc[base_idx]
          img_path = self.img_dir / f"subhalo_{row['subhalo_id']}.png"
          label = int(row["label"])
          image = Image.open(img_path).convert("RGB")

          rotation = self.rotations[aug_idx % 4]
          flip = (aug_idx // 4) == 1

          image = image.rotate(rotation)
          if flip:
              image = image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
          row = self.df.iloc[idx]
          img_path = self.img_dir / f"subhalo_{row['subhalo_id']}.png"

          image = Image.open(img_path).convert("RGB")
          label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


class CNN(nn.Module):
    def __init__(self, num_classes=3, feature_dim=128):
        super().__init__()

        # Convolutional feature extractor (unchanged)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Size-agnostic pooling → fixed-length embedding
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # (B, 128, 1, 1) for any HxW
        self.feature_dim = feature_dim          # = 128 after the conv stack

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        fmaps = self.features(x)                # (B, 128, H', W')
        z = torch.flatten(self.avgpool(fmaps), 1)  # (B, 128) regardless of input size
        out = self.classifier(z)
        return out, z
