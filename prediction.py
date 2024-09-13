import os
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, Normalize, Compose
from torchvision.io import read_image
from torchvision.models.video import r3d_18
from torch.amp import autocast

# Constants
DATA_DIR = 'data/raw_frames'
ANNOTATIONS_DIR = 'data/cvb_in_ava_format'
TRAIN_CSV = 'train.csv'
MODEL_SAVE_PATH = 'i3d_model.pth'
IMG_HEIGHT, IMG_WIDTH = 150, 150
NUM_CLASSES = 11
MAX_FRAMES = 450

# Data transforms
transform = Compose([
    Resize((IMG_HEIGHT, IMG_WIDTH)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

        return clip, label, video_name, cow_id

class I3D(nn.Module):
    def __init__(self, num_classes):
        super(I3D, self).__init__()
        self.model = r3d_18(weights="KINETICS400_V1")
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        # Permute x to [batch_size, channels, num_frames, height, width]
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)

# Function to load the dataset
def load_dataset():
    train_dataset = VideoDataset(os.path.join(ANNOTATIONS_DIR, TRAIN_CSV), transform=transform)
    return train_dataset

# Function to load the model
def load_model(model_path, device):
    model = I3D(num_classes=NUM_CLASSES)
    state_dict = torch.load(model_path, weights_only=True)

    # Handle models saved with DataParallel
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and model
    train_dataset = load_dataset()
    print(f'Length of train_dataset: {len(train_dataset)}')
    model = load_model(MODEL_SAVE_PATH, device)

    # Select a random training sample
    random_idx = random.randint(0, len(train_dataset) - 1)
    clip, true_label, video_name, cow_id = train_dataset[random_idx]

    # Log video name and cow ID
    print(f'Predicting for video: {video_name}, cow ID: {cow_id}')

    # Add batch dimension and move to device
    clip = clip.unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        with autocast(device_type='cuda'):
            outputs = model(clip)
            _, predicted_label = torch.max(outputs, 1)

    # Print true and predicted labels
    print(f'True Label: {true_label}, Predicted Label: {predicted_label.item()}')

if __name__ == '__main__':
    main()
