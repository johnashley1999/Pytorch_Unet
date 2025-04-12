import os
import shutil
import random

# Set base paths
base_path = "/home/john/Documents/Thesis/Pet_Pytorch_Unet/data"
image_folder = os.path.join(base_path, "images")
trimap_folder = os.path.join(base_path, "annotations/trimaps")

# Output directories
output_base = os.path.join(base_path, "split_dataset")
train_dir = os.path.join(output_base, "train")
valid_dir = os.path.join(output_base, "valid")
test_dir = os.path.join(output_base, "test")

# Subdirectories for images and trimaps
output_images = os.path.join(test_dir, "output")
train_images = os.path.join(train_dir, "images")
train_trimaps = os.path.join(train_dir, "trimaps")
valid_images = os.path.join(valid_dir, "images")
valid_trimaps = os.path.join(valid_dir, "trimaps")
test_images = os.path.join(test_dir, "images")
test_trimaps = os.path.join(test_dir, "trimaps")

os.makedirs(output_images, exist_ok=True)

# Create directories if they don’t exist
for dir_path in [train_images, train_trimaps, valid_images, valid_trimaps, test_images, test_trimaps]:
    os.makedirs(dir_path, exist_ok=True)

# Get list of image files (without extension to match with trimaps)
image_files = [f.split(".")[0] for f in os.listdir(image_folder) if f.endswith(".jpg")]
total_files = len(image_files)
print(f"Total files found: {total_files}")

# Define split ratios (e.g., 70% train, 15% valid, 15% test)
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15  # Should sum to 1.0

# Calculate number of files for each set
train_count = int(total_files * train_ratio)
valid_count = int(total_files * valid_ratio)
test_count = total_files - train_count - valid_count  # Ensure all files are used

print(f"Train: {train_count}, Valid: {valid_count}, Test: {test_count}")

# Randomly shuffle the list
random.shuffle(image_files)

# Split the files
train_files = image_files[:train_count]
valid_files = image_files[train_count:train_count + valid_count]
test_files = image_files[train_count + valid_count:]

# Function to copy files to destination
def copy_files(file_list, src_img_dir, src_trimap_dir, dst_img_dir, dst_trimap_dir):
    for base_name in file_list:
        img_src = os.path.join(src_img_dir, base_name + ".jpg")
        trimap_src = os.path.join(src_trimap_dir, base_name + ".png")
        img_dst = os.path.join(dst_img_dir, base_name + ".jpg")
        trimap_dst = os.path.join(dst_trimap_dir, base_name + ".png")
        
        if os.path.exists(img_src) and os.path.exists(trimap_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(trimap_src, trimap_dst)
        else:
            print(f"Warning: Missing file - Image: {img_src}, Trimap: {trimap_src}")

# Copy files to respective folders
print("Copying train files...")
copy_files(train_files, image_folder, trimap_folder, train_images, train_trimaps)

print("Copying validation files...")
copy_files(valid_files, image_folder, trimap_folder, valid_images, valid_trimaps)

print("Copying test files...")
copy_files(test_files, image_folder, trimap_folder, test_images, test_trimaps)

print("Dataset split complete!")
print(f"Train: {len(os.listdir(train_images))} images")
print(f"Valid: {len(os.listdir(valid_images))} images")
print(f"Test: {len(os.listdir(test_images))} images")