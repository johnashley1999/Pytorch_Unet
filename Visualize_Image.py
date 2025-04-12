from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2

# Set paths
project_folder = "/home/john/Documents/Thesis/Pet_Pytorch_Unet/data" #"my_project"  # Adjust as needed
image_folder = os.path.join(project_folder, "annotations/trimaps")
image_file = "Abyssinian_1.png"  # Change to any image in the dataset
image_path = os.path.join(image_folder, image_file)

# Check if the file exists
if os.path.exists(image_path):

    image = cv2.imread(image_path)

    if image is not None:

        scaled_image = (image -1) * 127

        cv2.imshow("Visualizing: " + image_file, scaled_image)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    else:
        print(f"error: image is none")
