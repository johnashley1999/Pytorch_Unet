import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Set random seed for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
# IMG_HEIGHT = 256
# IMG_WIDTH = 256
NUM_CLASSES = 3  # Foreground (pet), background, boundary
BATCH_SIZE = 32
# BATCH_SIZE = 16
# NUM_EPOCHS = 30
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# -------------------
# Dataset Class
# -------------------
class OxfordIIITPetDataset(Dataset):
    def __init__(self, root_dir, split="trainval", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Paths to images and trimaps based on split
        self.images_dir = os.path.join(root_dir, split, "images")
        self.masks_dir = os.path.join(root_dir, split, "trimaps")

        # List all image files (without extensions)
        self.image_names = [f.split(".")[0] for f in os.listdir(self.images_dir) if f.endswith(".jpg")]

        if not self.image_names:
            raise ValueError(f"No images found in {self.images_dir}. Check your dataset structure.")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, f"{img_name}.jpg")
        image = Image.open(img_path).convert("RGB")

        # Load mask (trimap)
        mask_path = os.path.join(self.masks_dir, f"{img_name}.png")
        mask = Image.open(mask_path).convert("L")  # Grayscale

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            # Separate mask transformation
            mask = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))(mask)
            mask = transforms.ToTensor()(mask) * 255  # Scale back to original values
            mask = mask.squeeze() - 1  # Remove channel dim, adjust to 0, 1, 2
            mask = mask.long()

        return image, mask

# -------------------
# Depth-wise Separable Convolution
# -------------------
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# -------------------
# SegNet Model
# -------------------
class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=NUM_CLASSES):
        super(SegNet, self).__init__()

        # Encoder
        self.enc1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc1_2 = DepthwiseSeparableConv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc2_1 = DepthwiseSeparableConv2d(64, 128, kernel_size=3, padding=1)
        self.enc2_2 = DepthwiseSeparableConv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc3_1 = DepthwiseSeparableConv2d(128, 256, kernel_size=3, padding=1)
        self.enc3_2 = DepthwiseSeparableConv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enc4_1 = DepthwiseSeparableConv2d(256, 512, kernel_size=3, padding=1)
        self.enc4_2 = DepthwiseSeparableConv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Bottleneck
        self.bottleneck = DepthwiseSeparableConv2d(512, 512, kernel_size=3, padding=1)

        # Decoder
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec4_1 = DepthwiseSeparableConv2d(512, 256, kernel_size=3, padding=1)
        self.dec4_2 = DepthwiseSeparableConv2d(256, 256, kernel_size=3, padding=1)

        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec3_1 = DepthwiseSeparableConv2d(256, 128, kernel_size=3, padding=1)
        self.dec3_2 = DepthwiseSeparableConv2d(128, 128, kernel_size=3, padding=1)

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec2_1 = DepthwiseSeparableConv2d(128, 64, kernel_size=3, padding=1)
        self.dec2_2 = DepthwiseSeparableConv2d(64, 64, kernel_size=3, padding=1)

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec1_1 = DepthwiseSeparableConv2d(64, 64, kernel_size=3, padding=1)
        self.dec1_2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.enc1_1(x))
        x = F.relu(self.enc1_2(x))
        x, idx1 = self.pool1(x)

        x = F.relu(self.enc2_1(x))
        x = F.relu(self.enc2_2(x))
        x, idx2 = self.pool2(x)

        x = F.relu(self.enc3_1(x))
        x = F.relu(self.enc3_2(x))
        x, idx3 = self.pool3(x)

        x = F.relu(self.enc4_1(x))
        x = F.relu(self.enc4_2(x))
        x, idx4 = self.pool4(x)

        # Bottleneck
        x = F.relu(self.bottleneck(x))

        # Decoder
        x = self.unpool4(x, idx4)
        x = F.relu(self.dec4_1(x))
        x = F.relu(self.dec4_2(x))

        x = self.unpool3(x, idx3)
        x = F.relu(self.dec3_1(x))
        x = F.relu(self.dec3_2(x))

        x = self.unpool2(x, idx2)
        x = F.relu(self.dec2_1(x))
        x = F.relu(self.dec2_2(x))

        x = self.unpool1(x, idx1)
        x = F.relu(self.dec1_1(x))
        x = self.dec1_2(x)

        return x

# -------------------
# Data Loading
# -------------------
# Define transforms
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

# Load dataset
# Note: You need to download the Oxford-IIIT Pet dataset and place it in the 'dataset' directory
# Expected structure: dataset/oxford-iiit-pet/images/, dataset/oxford-iiit-pet/annotations/
dataset_root = "/home/john/Documents/Thesis/Pet_Pytorch_Unet/data/split_dataset"
output_root = os.path.join(dataset_root, "test/output")
train_dataset = OxfordIIITPetDataset(root_dir=dataset_root, split="train", transform=transform)
test_dataset = OxfordIIITPetDataset(root_dir=dataset_root, split="test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# -------------------
# Model, Loss, and Optimizer
# -------------------
model = SegNet(in_channels=3, out_channels=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------
# Training Loop
# -------------------
print(f"Starting training for {NUM_EPOCHS} epochs...")
model.train()
total_step = len(train_loader)
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    running_loss = 0.0

    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}")

    epoch_time = time.time() - epoch_start_time
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] completed in {epoch_time:.2f} seconds, Average Loss: {running_loss / total_step:.4f}")

# Save the model
torch.save(model.state_dict(), "segnet_pets.pth")
print("Model saved as 'segnet_pets.pth'")

# -------------------
# Evaluation and Visualization
# -------------------
model.eval()
with torch.no_grad():
    # Get a batch from the test set
    images, masks = next(iter(test_loader))
    images = images.to(device)
    masks = masks.to(device)

    # Predict
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    # Convert to numpy for visualization
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    predicted = predicted.cpu().numpy()

    # Visualize the first image in the batch
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(images[0], (1, 2, 0)))
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(masks[0], cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(predicted[0], cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    # Save the visualization
    plt.savefig(output_root)
    plt.close()
    print("Visualization saved as 'test/output/predicted_mask.png'")

# -------------------
# Compute IoU
# -------------------
def compute_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float("nan"))  # Skip if no ground truth for this class
        else:
            ious.append(intersection / union)
    return ious

# Compute IoU on the test set
total_iou = [0.0] * NUM_CLASSES
num_batches = 0

model.eval()
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        iou = compute_iou(predicted, masks, NUM_CLASSES)
        for cls in range(NUM_CLASSES):
            if not np.isnan(iou[cls]):
                total_iou[cls] += iou[cls]
        num_batches += 1

# Average IoU
mean_iou = [iou / num_batches for iou in total_iou]
print("Mean IoU per class:", mean_iou)
print("Average IoU:", np.nanmean(mean_iou))

