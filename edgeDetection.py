import cv2
import numpy as np
from matplotlib import pyplot as plt


def cannyEdgeDetection(image):
    """Extracts canny edges from image / video frame."""
    color = False
    if len(image.shape) == 3:
        color = True

    if color:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection using Canny edge detector
    low_edge_threshold = 150
    high_edge_threshold = 250
    edges = cv2.Canny(gray, low_edge_threshold, high_edge_threshold, apertureSize=3, L2gradient=False)

    if color:
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return edges


def dollarsEdgeDetection(image):
    """Extracts dollars edges from image / video frame."""
    # Load pre-trained edge detection model
    model_path = "model.yml.gz"  # Download from OpenCV contrib
    edge_detector = cv2.ximgproc.createStructuredEdgeDetection(model_path)

    # Convert to RGB (Structured Edge Detection requires RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect edges
    edges = edge_detector.detectEdges(np.float32(rgb_image) / 255.0)

    return edges


def sobelEdgeDetection(image):
    """Extracts sobel edges from image / video frame."""
    # Compute Sobel gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Combine gradients
    edges = cv2.magnitude(sobelx, sobely) * 255.0

    # Normalize and display
    edges = np.uint8(255 * edges / np.max(edges))
    # # Prevent division by zero
    # if np.max(edges) > 0:
    #     edges = edges / np.max(edges)  # Normalize
    # else:
    #     edges = np.zeros_like(edges)  # Avoid NaNs

    # edges = np.uint8(edges * 255).astype(np.uint8)

    return edges


def increaseContrast(image):
    """Increases contrast for an image / video frame."""
    color = False
    if len(image.shape) == 3:
        color = True
    
    if color:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=25.0, tileGridSize=(8,8))
    edges = clahe.apply(gray)

    # Convert back to BGR for video output
    if color:
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return edges


def laplacianEdgeDetection(image):
    """Extracts edges via laplacian from image / video frame."""
    # color = False
    # if len(image.shape) == 3:
    #     color = True
    
    # if color:
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (3,3), 0)

    # Compute Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Convert to uint8
    edges = np.uint8(np.abs(laplacian))

    # Convert back to BGR for video output
    # if color:
    #     edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return edges


# def hedEdgeDetection(image_path):
#     # Load the pre-trained HED model
#     prototxt_path = "deploy.prototxt"
#     caffemodel_path = "hed_pretrained_bsds.caffemodel"

#     net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

#     # Load and preprocess the image
#     image = cv2.imread(image_path)
#     height, width = image.shape[:2]

#     # Convert image to blob format (required for the network)
#     blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(width, height),
#                                 mean=(104.00698793, 116.66876762, 122.67891434),
#                                 swapRB=False, crop=False)

#     # Set input and perform forward pass
#     net.setInput(blob)
#     edges = net.forward()

#     # Convert output to 8-bit and normalize
#     edges = edges[0, 0]  # Remove unnecessary dimensions
#     edges = (255 * (edges - edges.min()) / (edges.max() - edges.min())).astype("uint8")

#     # Show results
#     cv2.imshow("HED Edge Detection", edges)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def colorStructuralEdges(image):
#     edges = image
#     # Apply Gaussian blur to reduce noise
#     # edges = cv2.GaussianBlur(edges, (11,11), 5)

#     # Compute Sobel gradients in X and Y direction
#     sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
#     sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)

#     # Compute gradient magnitude
#     gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

#     # Normalize to range [0, 255]
#     gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
#     gradient_magnitude = gradient_magnitude.astype(np.uint8)

#     # # Apply thresholding to keep only strong edges
#     # _, edges = cv2.threshold(gradient_magnitude, 100, 255, cv2.THRESH_BINARY)

#     # Apply double threshold (hysteresis)
#     low_thresh, high_thresh = 100, 120
#     edges_strong = gradient_magnitude > high_thresh  # Strong edges
#     edges_weak = (gradient_magnitude >= low_thresh) & (gradient_magnitude <= high_thresh)  # Weak edges

#     # Combine edges
#     final_edges = np.uint8(edges_strong * 255)
#     edges = final_edges

#     edges = getStructuralEdges(image, image)

#     cv2.imshow('Structural Edges', edges)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()







# # Apply Felzenszwalb segmentation
# blurred = cv2.GaussianBlur(image, (5,5), 0)
# segments = felzenszwalb(blurred, scale=200, sigma=0.8, min_size=800)

# # Convert segments into an edge map
# edges = np.zeros_like(image)

# edges[:-1, :][segments[:-1,:] != segments[1:, :]] = [0, 255, 0]  # Horizontal edges
# edges[:, :-1][segments[:, :-1] != segments[:, 1:]] = [0, 255, 0]  # Vertical edges

# # Display result
# cv2.imshow('Segmented Structural Edges', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



def waterShedEdgeDetection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find sure background using dilation
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)

    # Compute distance transform
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Convert to integer
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply Watershed transform
    cv2.watershed(image, markers)

    # Mark edges
    edges = np.zeros_like(image)
    edges[markers == -1] = [0, 0, 255]  # Mark boundaries in red

    return edges


def differenceOfGaussiansEdgeDetection(image):
    blur1 = cv2.GaussianBlur(image, (3,3), 21)
    blur2 = cv2.GaussianBlur(image, (21, 21), 5)

    difference_of_gaussians = blur1 - blur2

    difference_of_gaussians = cv2.normalize(difference_of_gaussians, None, 0, 255, cv2.NORM_MINMAX)
    difference_of_gaussians = np.uint8(difference_of_gaussians)

    edges = cv2.Canny(difference_of_gaussians, 50, 100)

    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) # close small gaps
    edges = cv2.dilate(edges, kernel, iterations=1) # Thicken edges slightly

    return edges

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=250, minLineLength=150, maxLineGap=4)

    # Draw detected lines on the image
    line_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 1) # Draw colored lines

    edges = line_img

    return edges


# ==================================================================================================== #
# ==================================================================================================== #

# if __name__ == '__main__':
#     # image_path = "images/crosswalk1.jpg"
#     # image_path = "images/crosswalk2.jpg"
#     # image_path = "images/crosswalk3.png"
#     image_path = "images\\city1.png"
#     image = cv2.imread(image_path)
#     # edges = cannyEdgeDetection(image)
#     # edges = dollarsEdgeDetection(image)
#     # edges = hedEdgeDetection(image)
#     # edges = sobelEdgeDetection(image)
#     # edges = laplacianEdgeDetection(image)
#     # edges = waterShedEdgeDetection(image)
#     edges = differenceOfGaussiansEdgeDetection(image)



#     cv2.imshow("edges", edges)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # structural = getStructuralEdges(cv2.imread(image_path),edges)
#     # colorStructuralEdges(cv2.imread(image_path))




