import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

IMG_HEIGHT = 128
IMG_WIDTH = 128

class OxfordIIITPetDataset(Dataset):
    def __init__(self, root_dir, split="trainval", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.images_dir = os.path.join(root_dir, split, "images")
        self.masks_dir = os.path.join(root_dir, split, "trimaps")

        self.image_names = [f.split(".")[0] for f in os.listdir(self.images_dir) if f.endswith(".jpg")]
        if not self.image_names:
            raise ValueError(f"No images found in {self.images_dir}. Check your dataset structure.")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, f"{img_name}.jpg")
        mask_path = os.path.join(self.masks_dir, f"{img_name}.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))(mask)
            mask = transforms.ToTensor()(mask) * 255
            mask = mask.squeeze() - 1
            mask = mask.long()

        return image, mask
