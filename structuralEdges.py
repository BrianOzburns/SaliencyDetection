import os
import sys
from PIL import Image
import numpy as np
import cv2

from edgeDetection import sobelEdgeDetection


def plotEdges(edges, structural=None):
    """Provides an easy method to view transformations to each image / video frame."""
    # cv2.imshow('Original Sobel', edges)
    cv2.imshow("edges", edges)
    if structural is not None:
        # cv2.imshow('Longest Edges', structural)
        cv2.imshow("structural", structural)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getLongestEdges(image):
    """Structural edge detection method that looks for the longest edges."""
    edges = sobelEdgeDetection(image)

    # Convert the edges to grayscale
    edges_gray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to extract stronger edges (optional)
    _, sobel_thresh = cv2.threshold(edges_gray, 50, 255, cv2.THRESH_BINARY)

    # Morphological closing to remove small edges
    kernel = np.ones((9, 9), np.uint8)
    morphed = cv2.morphologyEx(sobel_thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours of the edges
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw the longest edges
    long_edges = np.zeros_like(image)

    # Filter and draw only the longest contours
    min_length = 150   # Adjust this value as needed
    for contour in contours:
        if cv2.arcLength(contour, closed=False) > min_length:
            cv2.drawContours(long_edges, [contour], -1, (255, 255, 255), thickness=1)

    return long_edges


def getStructuralEdges(image):
    """Extracts structural edges from image / video frame."""
    # Apply bilateral filtering
    filtered = cv2.bilateralFilter(image, d=50, sigmaColor=150, sigmaSpace=150)

    filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Reshape image to a 2D array of pixels
    pixels = filtered.reshape((-1, 3))

    # Apply K-means clustering
    K = 2  # Number of color segments
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(np.float32(pixels), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to integer (color values)
    centers = np.uint8(centers)
    segmented_filtered = centers[labels.flatten()]  # Map each pixel to cluster center
    segmented_filtered = segmented_filtered.reshape(filtered.shape)  # Reshape back to image dimensions

    blurred = segmented_filtered
    # blurred = cv2.GaussianBlur(blurred, (5,5), 3)

    # Detect edges using Sobel
    edges_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = sobelEdgeDetection(edges_gray)
    # gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    # edges = cv2.Canny(gray, 100, 100)

    return edges

    # # Convert edges to 3-channel (so it overlays properly)
    # edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # return edges_colored

    # # Overlay edges on the segmented image
    # overlay = cv2.addWeighted(segmented_filtered, 0.8, edges_colored, 0.5, 0)

    # return overlay


def extractStructuralEdges(image):
    """Extracts structural edges from image / video frame."""

    # img = Image.fromarray(image)
    # img.save(os.path.join("image_outputs","original.png"))

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # img = Image.fromarray(gray)
    # img.save(os.path.join("image_outputs","grayscale.png"))

    # Apply Gaussian blur to reduce finer details and noise
    # blurred = cv2.GaussianBlur(gray, (7, 7), 2)

    # Apply Bilateral Filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, d=11, sigmaColor=75, sigmaSpace=75)

    # plotEdges(filtered)
    # img = Image.fromarray(filtered)
    # img.save(os.path.join("image_outputs","bilateral_filtered.png"))

    # Apply Canny edge detection with adjusted thresholds
    edges = cv2.Canny(filtered, threshold1=25, threshold2=150)

    # plotEdges(edges)
    # img = Image.fromarray(edges)
    # img.save(os.path.join("image_outputs","canny_edges.png"))

    # Use dilation to make the main edges thicker and reduce fine details
    dilated_edges = cv2.dilate(edges, None, iterations=3)

    # plotEdges(dilated_edges)
    # img = Image.fromarray(dilated_edges)
    # img.save(os.path.join("image_outputs","dilated_edges.png"))

    # Optional: Use erosion to remove small artifacts and refine larger edges
    refined_edges = cv2.erode(dilated_edges, None, iterations=3)

    # plotEdges(refined_edges)
    # img = Image.fromarray(refined_edges)
    # img.save(os.path.join("image_outputs","refined_edges.png"))

    # Apply Gaussian blur to reduce finer details and noise
    # blurred = cv2.GaussianBlur(refined_edges, (31, 31), 1)

    # plotEdges(blurred)

    # Morphological closing to connect broken edges and remove noise
    kernel = np.ones((3, 3), np.uint8)
    closed_edges = cv2.morphologyEx(refined_edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    # plotEdges(closed_edges)
    # img = Image.fromarray(closed_edges)
    # img.save(os.path.join("image_outputs","closed_edges.png"))

    # Find contours on the edge image
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank canvas to draw the largest contours (structural edges)
    edge_canvas = np.zeros_like(image)

    # Draw only the largest contours (keep only the structural blocks)
    for contour in contours:
        # # Approximate contours to simplify jagged edges
        # epsilon = 0.0000001 * cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, epsilon, True)

        # Filter out small contours based on area (this keeps the larger structural edges)
        if cv2.contourArea(contour) > 500:  # Minimum contour area threshold to keep large structures
            cv2.drawContours(edge_canvas, [contour], -1, (0, 255, 0), 2)  # Draw green for structural lines

    # Convert the edge_canvas to grayscale to make it easier to view
    edge_canvas_gray = cv2.cvtColor(edge_canvas, cv2.COLOR_BGR2GRAY)

    # plotEdges(edge_canvas_gray)
    # img = Image.fromarray(edge_canvas_gray)
    # img.save(os.path.join("image_outputs","contour_edges.png"))

    # Apply a threshold to only keep the strongest edges (removes low-intensity noise)
    _, thresholded_edges = cv2.threshold(edge_canvas_gray, 1, 40, cv2.THRESH_BINARY)
    
    final_edges = cv2.cvtColor(thresholded_edges, cv2.COLOR_GRAY2BGR)

    # plotEdges(final_edges)
    # img = Image.fromarray(final_edges)
    # img.save(os.path.join("image_outputs","final_edges.png"))

    return final_edges


def extractUnnoisyEdges(image):
    """Extracts structural edges from image / video frame by attempting to reduce noise as much as possible."""
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce finer details and noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)

    # Apply Canny edge detection with adjusted thresholds
    edges = cv2.Canny(blurred, threshold1=100, threshold2=250)

    # Use dilation to make the main edges thicker and reduce fine details
    dilated_edges = cv2.dilate(edges, None, iterations=3)

    # Optional: Use erosion to remove small artifacts and refine larger edges
    refined_edges = cv2.erode(dilated_edges, None, iterations=2)

    # Find contours on the edge image
    contours, _ = cv2.findContours(refined_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank canvas to draw the largest contours (structural edges)
    edge_canvas = np.zeros_like(image)

    # Draw only the largest contours (keep only the structural blocks)
    for contour in contours:
        # Filter out small contours based on area (this keeps the larger structural edges)
        if cv2.contourArea(contour) > 250:  # Minimum contour area threshold to keep large structures
            cv2.drawContours(edge_canvas, [contour], -1, (0, 255, 0), 2)  # Draw green for structural lines

    # Convert the edge_canvas to grayscale to make it easier to view
    edge_canvas_gray = cv2.cvtColor(edge_canvas, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to only keep the strongest edges (removes low-intensity noise)
    _, thresholded_edges = cv2.threshold(edge_canvas_gray, 1, 255, cv2.THRESH_BINARY)
    return thresholded_edges


def preprocess_image(image):
    """Load and preprocess the image (convert color spaces, apply blur)."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    bilateral_filtered = cv2.bilateralFilter(image, d=11, sigmaColor=25, sigmaSpace=25)
    return bilateral_filtered, hsv, lab


def color_segmentation(image, k=2):
    """Segment the image using k-means clustering."""
    pixel_vals = image.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    
    # Define criteria and apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(image.shape)
    return segmented_image


def detect_edges(image, min_thresh, max_thresh):
    """Apply Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, min_thresh, max_thresh)  # Tune thresholds based on needs
    return edges


def refine_edges(edges):
    """Refine edges while preserving structural integrity."""
    if len(edges.shape) == 3:
        edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding to enhance edges
    adaptive_edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 1)

    # Use a kernel for slight dilation while maintaining shapes
    kernel = np.array([[0, 1, 0], 
                       [1, 1, 1], 
                       [0, 1, 0]], dtype=np.uint8)
    
    refined = cv2.dilate(adaptive_edges, kernel, iterations=1)  # Small dilation to connect edges

    return refined


def filter_contours(edges):
    """Remove small contours to keep only major structural edges."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # Keep only large structures (adjust threshold)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=1)

    return mask


def segment_objects(image, k=3):
    """Segment main objects using k-means clustering."""
    pixel_vals = image.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    # Define criteria and apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(image.shape)

    return segmented_image


def create_mask(segmented_image):
    """Convert the segmented image into a binary mask."""
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def detect_edges_from_mask(mask):
    """Apply edge detection on the segmented mask."""
    return cv2.Canny(mask, 50, 150)


def structuralEdgesMethod2(image):
    image, hsv, lab = preprocess_image(image)
    segmented = color_segmentation(image)
    # edges = detect_edges(segmented, 50, 150)
    refined_edges = refine_edges(segmented)
    colored_edges = cv2.cvtColor(refined_edges, cv2.COLOR_GRAY2BGR)
    structural_edges = detect_edges(colored_edges, 400, 500)
    final_edges = filter_contours(structural_edges)

    # plotEdges(image)
    # plotEdges(segmented)
    # # plotEdges(edges)
    # plotEdges(refined_edges)
    # plotEdges(structural_edges)
    # plotEdges(cv2.subtract(edges,final_edges))

    final_edges = cv2.cvtColor(final_edges, cv2.COLOR_GRAY2BGR)

    return final_edges


def testing(image):
    # # Step 1: Segment the main objects
    # segmented = segment_objects(image)

    # # Step 2: Create a binary mask
    # mask = create_mask(segmented)

    # # Step 3: Apply edge detection on the mask
    # edges = detect_edges_from_mask(mask)

    # plotEdges(edges)

    edges = structuralEdgesMethod2(image)

    # unnoisyEdges = extractUnnoisyEdges(image)

    # plotEdges(unnoisyEdges)


# image_paths = [os.path.join("images","city1.png"),os.path.join("images","city2.png"),os.path.join("images","city3.png"),os.path.join("images","city4.png")]
# for image_path in image_paths:
#     image = cv2.imread(image_path)
#     testingMethod(image)

# Tried canny with long edge filtering
# Tried HoughLines
# Tried Difference of Gaussians
# Tried pretrained structure detecting model
# Tried Watershed
# Tried felzenszwalb





