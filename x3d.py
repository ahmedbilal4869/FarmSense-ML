import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# Constants
DATA_DIR = 'data/raw_frames'
ANNOTATIONS_DIR = 'data/cvb_in_ava_format'
TRAIN_CSV = 'train.csv'
VAL_CSV = 'val.csv'
IMG_HEIGHT, IMG_WIDTH = 256, 256
NUM_CLASSES = 11
BATCH_SIZE = 4
EPOCHS = 10
MAX_FRAMES = 450  # or any other value based on your dataset

MODEL_SAVE_PATH = 'x3d_model.pth'
SAMPLE_FRAMES_DIR = 'sample_frames'

class VideoDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.annotations = pd.read_csv(
            annotations_file,
            skiprows=1,
            names=[
                "video_name",
                "keyframe",
                "x1",
                "y1",
                "x2",
                "y2",
                "behavior_category",
                "cow_id",
            ],
            dtype={
                "keyframe": float,
                "x1": float,
                "y1": float,
                "x2": float,
                "x2": float,
                "behavior_category": int,
                "cow_id": str,
            },
            low_memory=False,
        )
        self.transform = transform
        self.frame_counts = self._count_frames()
        self.clip_indices = self._create_clip_indices()

    def _count_frames(self):
        frame_counts = {}
        current_video_name = None
        current_cow_id = None
        count = 0

        for _, row in self.annotations.iterrows():
            video_name = row['video_name']
            cow_id = row['cow_id']
            if video_name == current_video_name and cow_id == current_cow_id:
                count += 1
            else:
                if current_video_name is not None:
                    frame_counts[(current_video_name, current_cow_id)] = count
                current_video_name = video_name
                current_cow_id = cow_id
                count = 1

        if current_video_name is not None:
            frame_counts[(current_video_name, current_cow_id)] = count

        return frame_counts

    def _create_clip_indices(self):
        clip_indices = []
        current_video_name = None
        current_cow_id = None

        for idx, row in self.annotations.iterrows():
            video_name = row['video_name']
            cow_id = row['cow_id']
            if video_name != current_video_name or cow_id != current_cow_id:
                clip_indices.append(idx)
                current_video_name = video_name
                current_cow_id = cow_id

        return clip_indices

    def __len__(self):
        return len(self.clip_indices)

    def __getitem__(self, idx):
        start_idx = self.clip_indices[idx]
        row = self.annotations.iloc[start_idx]
        video_name = row['video_name']
        cow_id = row['cow_id']
        keyframe = int(row['keyframe'])  # Ensure this conversion matches your frame rate
        x1, y1, x2, y2 = (row['x1'], row['y1'], row['x2'], row['y2'])
        label = int(row['behavior_category']) - 2  # Adjusting for zero-based index

        num_frames = self.frame_counts.get((video_name, cow_id), 0)
        frames = []

        # Ensure frames have the same length
        for i in range(MAX_FRAMES):
            frame_path = os.path.join(DATA_DIR, video_name, f'img_{i+1:05d}.jpg')
            if i < num_frames and os.path.exists(frame_path):
                image = read_image(frame_path).float() / 255.0  # Normalize to [0, 1]
                _, img_height, img_width = image.shape

                # Convert normalized bounding box to pixel coordinates
                x1_pixel = int(x1 * img_width)
                y1_pixel = int(y1 * img_height)
                x2_pixel = int(x2 * img_width)
                y2_pixel = int(y2 * img_height)

                # Ensure bounding box coordinates are within image dimensions
                x1_pixel = max(0, min(x1_pixel, img_width - 1))
                y1_pixel = max(0, min(y1_pixel, img_height - 1))
                x2_pixel = max(x1_pixel + 1, min(x2_pixel, img_width))
                y2_pixel = max(y1_pixel + 1, min(y2_pixel, img_height))

                if x2_pixel > x1_pixel and y2_pixel > y1_pixel:
                    cropped_image = image[:, y1_pixel:y2_pixel, x1_pixel:x2_pixel]
                else:
                    cropped_image = torch.zeros((3, IMG_HEIGHT, IMG_WIDTH))

                if cropped_image.shape[1] > 0 and cropped_image.shape[2] > 0:
                    if self.transform:
                        cropped_image = self.transform(cropped_image)
                else:
                    cropped_image = torch.zeros((3, IMG_HEIGHT, IMG_WIDTH))

                frames.append(cropped_image)
            else:
                frames.append(torch.zeros((3, IMG_HEIGHT, IMG_WIDTH)))

        # Pad with zeros if fewer frames
        if len(frames) < MAX_FRAMES:
            frames.extend([torch.zeros((3, IMG_HEIGHT, IMG_WIDTH))] * (MAX_FRAMES - len(frames)))

        clip = torch.stack(frames)

        if label < 0 or label >= NUM_CLASSES:
            print(f"Invalid label {label} at index {idx}")

        return clip, label

    def get_num_clips(self):
        return len(self.clip_indices)

def save_clip_as_frames(clip, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, frame in enumerate(clip):
        frame_np = frame.permute(1, 2, 0).cpu().numpy()  # [height, width, channels]
        frame_np = (frame_np * 255).astype(np.uint8)  # Convert to uint8

        # Convert from RGB to BGR
        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        frame_path = os.path.join(output_dir, f'frame_{i+1:04d}.jpg')
        cv2.imwrite(frame_path, frame_np)

    print(f"Clip frames saved to {output_dir}")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load datasets
def load_datasets():
    train_dataset = VideoDataset(os.path.join(ANNOTATIONS_DIR, TRAIN_CSV), transform=transform)
    val_dataset = VideoDataset(os.path.join(ANNOTATIONS_DIR, VAL_CSV), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=18, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=18, pin_memory=True)

    return train_loader, val_loader, train_dataset, val_dataset

# Define the X3D model
class X3D(nn.Module):
    def __init__(self, num_classes):
        super(X3D, self).__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
        self.model.blocks[-1].proj = nn.Linear(self.model.blocks[-1].proj.in_features, num_classes)

    def forward(self, x):
        # Permute x to [batch_size, channels, num_frames, height, width]
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_loader, val_loader, train_dataset, val_dataset = load_datasets()

    # Log the number of clips in the train dataset
    num_clips = train_dataset.get_num_clips()
    print(f'Number of clips in the train dataset: {num_clips}')

    # Save a sample clip before training
    clip, label = train_dataset[0]  # Get the first clip
    save_clip_as_frames(clip, SAMPLE_FRAMES_DIR)

    # Initialize the model, loss function, and optimizer
    model = X3D(num_classes=NUM_CLASSES).to(device)
    
    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()
    
    torch.cuda.empty_cache()

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total

        print(f'Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print('Model saved!')

if __name__ == '__main__':
    main()
