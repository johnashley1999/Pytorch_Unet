import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from Unet_Model import SegNet
from Dataset import OxfordIIITPetDataset

# -------------------
# Configs
# -------------------
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 3
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Load Model
# -------------------
model = SegNet(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("segnet_pets.pth", map_location=DEVICE))
model.eval()
print("Loaded trained weights into SegNet model.")

# -------------------
# Define Transform
# -------------------
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

# -------------------
# Load Dataset
# -------------------
dataset_root = "/home/john/Documents/Thesis/Pet_Pytorch_Unet/data/split_dataset"
output_dir = os.path.join(dataset_root, "test/inference_outputs")
os.makedirs(output_dir, exist_ok=True)

test_dataset = OxfordIIITPetDataset(root_dir=dataset_root, split="test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------
# Inference and Visualization (10 images)
# -------------------
with torch.no_grad():
    for i, (images, masks) in enumerate(test_loader):
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

        save_path = os.path.join(output_dir, f"predicted_mask_{i}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Visualization saved as '{save_path}'")

        if i == 9:
            break