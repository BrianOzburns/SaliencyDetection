# Load in the video
# Extract the frames
# For the first frame, ask gpt for the rankings
# After gpt has the rankings, execute YOLO on the rest of the frames
# Save the YOLO frames
# Combine the YOLO frames into a video

import os
import openai
import sys
import base64
import json
from io import BytesIO
from PIL import Image
from collections import defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv

# from google.colab.patches import cv2_imshow
# from cv2_plt_imshow import cv2_plt_imshow

from ultralytics import YOLO
import torch

from edgeDetection import sobelEdgeDetection, increaseContrast
from structuralEdges import getStructuralEdges, extractStructuralEdges, structuralEdgesMethod2, extractUnnoisyEdges



GPT_MODEL="gpt-4o"


def plotEdges(edges, structural=None):
    """Provides an easy method to view transformations to each image / video frame."""
    cv2.imshow("Edges", edges)
    if structural is not None:
        cv2.imshow("Structural Edges", structural)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def encodeImage(image, format="PNG"):
    """Converts a PIL image to a base64-encoded string."""
    buffered = BytesIO()
    image.save(buffered, format=format)  # Save the image to a buffer
    return base64.b64encode(buffered.getvalue()).decode()  # Encode to base64


def analyzeScene(image_base64, api_key):
    """Analyzes an image and returns a ranked description."""

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)

    # OpenAI API request
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are an AI that assists visually-impaired individuals by analyzing scenes."},
            {"role": "user", "content": [
                {"type": "text", "text": (
                    "This image is from my point of view. "
                    "In one sentence, describe to me the setting. "
                    "In one sentence, describe to me what I am doing. "
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content


def rankItems(frame, api_key):
    """Analyzes an image and returns a ranked description."""
    # labels from yolo vocabulary
    labels = "person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush"

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)

    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Encode image in base64 format
    image_base64 = encodeImage(image_pil)

    # Get context
    context = analyzeScene(image_base64, api_key)

    # OpenAI API request
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are an AI that assists visually-impaired individuals by analyzing scenes. "
                "Always respond in the following JSON format: {\"items\": [\"item1\", \"item2\", ...]}"
            )},
            {"role": "assistant", "content": context},
            {"role": "assistant", "content": "all returned items must exist in the following vocabulary: " + labels},
            {"role": "user", "content": [
                {"type": "text", "text": (
                    "I am a visually-impaired individual with a retinal implant. "
                    "I need to see only the most important information in the scene and do not want to be distracted by what is unimportant. "
                    "Based on what I seem to be doing, rank the 20 most significant physical things in the scene. "
                    "Consider that large or potentially fast moving objects are likely more significant in scenes rather than smaller objects. "
                    "As an example, cars are more significant for a visually-impaired indiviual than backpacks."
                    "People are important to me."
                    "Reason about the context of the scene, and consider what I would want to see in the scene. Ask yourself 'what are the most important objects in the scene?'."
                    "All returned objects must exist in the given vocabulary:" + labels + ". Make sure that this is the case."
                    "Return the results strictly as a JSON object with a 'items' field ordered by significance."
                    "Each item in the result must be unique."
                    "There should be 20 items in the final JSON object"

                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}
        ],
        response_format={ "type": "json_object" },
        max_tokens=500
    )

    print(response)

    result_json = json.loads(response.choices[0].message.content)
    result = result_json["items"]

    return result


def top_k_segments(ranking, result, k=4):
    """Selects the top k elements from  OpenAI ranking and YOLO objects."""
    all_class_segments = defaultdict(list)
    for d in result[0].summary():
        class_name = d['name']
        if class_name in all_class_segments:
            all_class_segments[class_name].append(d['segments'])
        else:
            all_class_segments[class_name] = [d['segments']]

    output_segment = []
    top_k = k
    for obj in ranking:
        if obj in all_class_segments:
            output_segment.extend(all_class_segments[obj])
            top_k -= 1
            if top_k == 0:
                break
    return output_segment


def getSalientOnlyImg(img, output_segment):
    """Grabs only the segmented objects in the top_k segments."""
    salient_only_img = np.zeros_like(img, dtype=np.uint8)
    for segment in output_segment:
        points_rollers = [(int(x), int(y)) for x, y in zip(segment['x'], segment['y'])]
        data_points = [tuple(point) for point in points_rollers]
        # Create a separate single-channel mask
        object_mask = np.zeros(img.shape[:2], dtype=np.uint8)  # Grayscale mask
        cv2.fillPoly(object_mask, [np.array(data_points)], 255)  # White polygon mask

        # Use the mask to extract pixels from the original image and add the extracted pixels to the final image
        masked_object = cv2.bitwise_and(img, img, mask=object_mask)
        salient_only_img = cv2.add(salient_only_img, masked_object)

    return salient_only_img


def main(args):
    assert(len(args) > 1)

    testing = False
    if args[1] == "testing" or args[1] == "Testing":
        testing = True

    if testing:
        image_paths = [os.path.join("images","city1.png"),os.path.join("images","city2.png"),os.path.join("images","city3.png"),os.path.join("images","city4.png")]
        for image_path in image_paths:
            image = cv2.imread(image_path)
            sobel_edges = sobelEdgeDetection(image)
            edges = increaseContrast(sobel_edges)
            plotEdges(edges)
            
            structural_edges = extractStructuralEdges(image)
            # structural_edges = getStructuralEdges(image)
            # structural_edges = get
            plotEdges(structural_edges)
        return 0
    
    video = args[1]
    output_video = video

    # video = "sunset-city-walk.mov"
    # output_video = "sunset-city-walk-k=2.mp4"

    # video = "walking-in-park.mov"
    # output_video = "walking-in-park=2.mp4"

    # video = "car1.mov"
    # output_video = "car1-k=2.mp4"

    # video = "car2.mov"
    # output_video = "car2-k=2.mp4"

    video_path = os.path.join("videos", video)
    output_video_path = os.path.join("video_outputs", output_video)

    model = os.path.join("models","yolov8m-seg.pt")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(model)
    model.to(device)

    # Open video capture
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for AVI format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    count = 0

    # Read video frames, process each frame, output transformed frame to out video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count == 1:
            # for the first frame in the video, generate the rankings
            # each frame is a numpy array, need to change function definitions for ranking
            # rankings = rankItems(image_paths[i] , openai.api_key)
            rankings = rankItems(frame, openai.api_key)
            print(rankings)
        # For faster inference, only look at every 10th frame
        # if count % 10 != 0:
        #    continue

        # Run YOLOv8 segmentation on the frame
        results = model(frame)

        # Get top K segments
        output_segment = top_k_segments(rankings, results, k=2)

        # Get salient only img
        salient_only_img = getSalientOnlyImg(frame, output_segment)

        # Get Sobel and Structural Edges
        # salient_only_img = cv2.cvtColor(salient_only_img, cv2.COLOR_BGR2GRAY)
        sobel_edges = sobelEdgeDetection(salient_only_img)
        edges = increaseContrast(sobel_edges)
        # all_img_edges = sobelEdgeDetection(frame)
        # structural_edges = getStructuralEdges(frame, all_img_edges)
        structural_edges = extractStructuralEdges(frame)
        # structural_edges = structuralEdgesMethod2(frame)

        print(edges.shape)
        print(structural_edges.shape)

        overlay = cv2.bitwise_or(edges, structural_edges)

        # Render the segmentation masks on the frame
        # annotated_frame = results[0].plot()  # Renders bounding boxes & masks

        # Write the processed frame to output video
        out.write(overlay)
        print(count)

    # Release resources
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':

    # Setup openai
    load_dotenv(find_dotenv()) # read local .env file
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = openai_api_key
    os.environ['OPENAI_API_KEY'] = openai_api_key

    main(sys.argv)

