import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import openai
from dotenv import load_dotenv, find_dotenv

from ultralytics import YOLO
import torch

from yoloVideo import top_k_segments, rankItems, getSalientOnlyImg, sobelEdgeDetection
from structuralEdges import extractStructuralEdges
from edgeDetection import increaseContrast


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

    # Run YOLOv8 segmentation on the frame
    image = cv2.imread(image_path)

    results = model(image)

    # Get top K segments
    rankings = rankItems(image, openai.api_key)
    output_segment = top_k_segments(rankings, results, k=4)

    # Get salient only img
    salient_only_img = getSalientOnlyImg(image, output_segment)
    color_adjusted = cv2.cvtColor(salient_only_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(color_adjusted)
    img.save(os.path.join("image_outputs","salient_only_img.png"))

    # Get Sobel and Structural Edges
    # salient_only_img = cv2.cvtColor(salient_only_img, cv2.COLOR_BGR2GRAY)
    sobel_edges = sobelEdgeDetection(salient_only_img)
    edges = increaseContrast(sobel_edges)
    # all_img_edges = sobelEdgeDetection(frame)
    # structural_edges = getStructuralEdges(frame, all_img_edges)
    structural_edges = extractStructuralEdges(image)
    # structural_edges = structuralEdgesMethod2(frame)
    
    overlay = cv2.bitwise_or(edges, structural_edges)
    img = Image.fromarray(overlay)
    img.save(os.path.join("image_outputs","final.png"))
    exit(0)


image_path = os.path.join("images","crosswalk2.jpg")

# Setup openai
load_dotenv(find_dotenv()) # read local .env file
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key

yoloOnImage(image_path)


# Encountering some python matplotlib issue when running

# visualizeSequentialImages(os.path.join("image_outputs", "city1", "structural"))
# visualizeSequentialImages(os.path.join("image_outputs", "city2", "structural"))
# visualizeSequentialImages(os.path.join("image_outputs", "city3", "structural"))
# visualizeSequentialImages(os.path.join("image_outputs", "city4", "structural"))
# visualizeSequentialImages(os.path.join("image_outputs", "crosswalk2", "structural"))
# visualizeSequentialImages(os.path.join("image_outputs", "concert", "structural"))

