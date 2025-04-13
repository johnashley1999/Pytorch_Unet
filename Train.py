import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from Dataset import OxfordIIITPetDataset
from Unet_Model import SegNet

# Configs
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 3
BATCH_SIZE = 32
NUM_EPOCHS = 350
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_root = "/home/john/Documents/Thesis/Pet_Pytorch_Unet/data/split_dataset"
output_root = os.path.join(dataset_root, "test/output")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

# Load data
train_dataset = OxfordIIITPetDataset(root_dir=dataset_root, split="train", transform=transform)
test_dataset = OxfordIIITPetDataset(root_dir=dataset_root, split="test", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Initialize model
model = SegNet(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.5)

# Training loop
print(f"Using device: {DEVICE}")
print(f"Starting training for {NUM_EPOCHS} epochs...")
total_step = len(train_loader)
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    epoch_start_time = time.time()

    for i, (images, masks) in enumerate(train_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}")

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] completed in {epoch_time:.2f}s, Avg Loss: {running_loss / total_step:.4f}, LR: {current_lr:.6f}")

# Save model
torch.save(model.state_dict(), "segnet_pets.pth")
print("Model saved as 'segnet_pets.pth'")

# Evaluate & visualize
model.eval()
with torch.no_grad():
    images, masks = next(iter(test_loader))
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    predicted = predicted.cpu().numpy()

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
    os.makedirs(output_root, exist_ok=True)
    plt.savefig(os.path.join(output_root, "predicted_mask.png"))
    plt.close()
    print(f"Saved visualization to {output_root}/predicted_mask.png")
