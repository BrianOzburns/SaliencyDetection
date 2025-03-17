import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2

from ultralytics import YOLO
import torch


def visualizeSequentialImages(directory):
    titles = ["original", "grayscale", "bilateral_filtered", "canny_edges", 
              "dilated_edges", "refined_edges", "closed_edges", "contour_edges"]
    image_names = [title + ".png" for title in titles]

    images = []
    valid_titles = []  # Store only titles of images that exist
    for title, image_name in zip(titles, image_names):
        image_path = os.path.join(directory, image_name)
        if os.path.exists(image_path):
            image = Image.open(image_path)
            images.append(image)
            valid_titles.append(title)  # Append title only if image exists

    num_images = len(images)
    if num_images == 0:
        print("No images found. Exiting.")
        return
    
    # Determine grid layout
    rows = 2
    cols = (num_images + 1) // 2  # Distribute images evenly into two rows

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Display images with titles
    for i, (img, title) in enumerate(zip(images, valid_titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis("off")  # Hide axes

    # Hide any unused subplots if images < expected slots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def yoloOnImage(image_path):
    model = os.path.join("models","yolov8m-seg.pt")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(model)
    model.to(device)

    image = cv2.imread(image_path)
    results = model(image)

    cv2.imshow("results", results)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = os.path.join("images","city1.png")
yoloOnImage(image_path)


# Encountering some python matplotlib issue when running

# visualizeSequentialImages(os.path.join("image_outputs", "city1", "structural"))
# visualizeSequentialImages(os.path.join("image_outputs", "city2", "structural"))
# visualizeSequentialImages(os.path.join("image_outputs", "city3", "structural"))
# visualizeSequentialImages(os.path.join("image_outputs", "city4", "structural"))
# visualizeSequentialImages(os.path.join("image_outputs", "crosswalk2", "structural"))
# visualizeSequentialImages(os.path.join("image_outputs", "concert", "structural"))

