import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm
from torchvision.models.video import r3d_18
from torch.cuda.amp import GradScaler, autocast

# Constants
DATA_DIR = 'data/raw_frames'
ANNOTATIONS_DIR = 'data/cvb_in_ava_format'
TRAIN_CSV = 'ava_train_set.csv'
VAL_CSV = 'ava_val_set.csv'
IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_CLASSES = 12
BATCH_SIZE = 1
EPOCHS = 1

class VideoDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.annotations = pd.read_csv(annotations_file, skiprows=1, names=["video_name", "keyframe", "x1", "y1", "x2", "y2", "behavior_category", "animal_category"], dtype={"keyframe": float, "x1": float, "y1": float, "x2": float, "y2": float, "behavior_category": int, "animal_category": str}, low_memory=False)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_name = self.annotations.iloc[idx, 0]
        keyframe = int(self.annotations.iloc[idx, 1] * 30)  # Convert seconds to frame index
        frames = []

        for i in range(450):
            frame_path = os.path.join(DATA_DIR, video_name, f'img_{i+1:05d}.jpg')
            if not os.path.exists(frame_path):
                continue  # Skip if frame file doesn't exist

            image = read_image(frame_path).float() / 255.0  # Normalize to [0, 1]
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        if len(frames) == 0:
            return None, None  # Return None if no valid frames found

        clip = torch.stack(frames)  # Shape: [num_frames, 3, 224, 224]
        label = int(self.annotations.iloc[idx, 6])  # behavior_category

        return clip, label

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load datasets
def load_datasets():
    train_dataset = VideoDataset(os.path.join(ANNOTATIONS_DIR, TRAIN_CSV), transform=transform)
    val_dataset = VideoDataset(os.path.join(ANNOTATIONS_DIR, VAL_CSV), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader

# Define the I3D model
class I3D(nn.Module):
    def __init__(self, num_classes):
        super(I3D, self).__init__()
        self.model = r3d_18(weights="KINETICS400_V1")
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        # Permute x to [batch_size, channels, num_frames, height, width]
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_loader, val_loader = load_datasets()

    # Initialize the model, loss function, and optimizer
    model = I3D(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for clips, labels in tqdm(train_loader):
            if clips is None:
                continue

            clips, labels = clips.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(clips)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader)}, Accuracy: {100.*correct/total}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for clips, labels in tqdm(val_loader):
                if clips is None:
                    continue

                clips, labels = clips.to(device), labels.to(device)

                with autocast():
                    outputs = model(clips)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f'Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {100.*correct/total}')

if __name__ == '__main__':
    main()
