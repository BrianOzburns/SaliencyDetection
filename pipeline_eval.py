# Load in the video
# Extract the frames
# For the first frame, ask gpt for the rankings
# After gpt has the rankings, execute YOLO on the rest of the frames
# Save the YOLO frames
# Combine the YOLO frames into a video

import os
import time

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

from skimage import io
from scipy.ndimage import gaussian_filter

from threading import Thread


GPT_MODEL = "gpt-4o"

class Worker(Thread):
    def __init__(self, id, fname_list: list[str]):
        super().__init__()
        self.id = id
        self.fname_list = fname_list
    def run(self):
        model_output_dir = os.path.join("eval_output_images")
        orig_imgs_dir = os.path.join("DUT_OMRON_dataset", "orig_images")

        model = os.path.join("models", "yolov8m-seg.pt")
        model = YOLO(model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        for image_name in self.fname_list:
            image_path = os.path.join(orig_imgs_dir, image_name)
            if os.path.exists(os.path.join(model_output_dir, image_name[:-4] + '.png')):
                print(f"Using existing pipeline output for {image_name}. Skipping.")
                continue
            image = cv2.imread(image_path)
            # Run half the pipeline and get the salient object masks
            try:
                rankings = rankItems(image, openai.api_key)
            except:
                print(f"Unstable API, skipping {image_name}. Rerun python pipeline_eval.py upon completion.")
                continue
            results = model(image)

            # Get top K segments
            output_segment = top_k_segments(rankings, results, k=2)

            # Get salient only img
            salient_only_img = getSalientOnlyImg(image, output_segment)
            cv2.imwrite(os.path.join(model_output_dir, image_name[:-4] + '.png'), salient_only_img)

def s_measure(pred_mask, gt_mask):
    """
    Compute the S-measure between a predicted mask and the ground truth mask.
    """
    # Ensure binary format (normalize between 0 and 1)
    pred_mask = pred_mask.astype(np.float32) / 255.0
    gt_mask = gt_mask.astype(np.float32) / 255.0

    # Region-aware structural similarity (S-region)
    gt_mean = np.mean(gt_mask)
    pred_mean = np.mean(pred_mask)
    s_region = 1 - np.abs(gt_mean - pred_mean)

    # Object-aware structural similarity (S-object)
    gt_blurred = gaussian_filter(gt_mask, sigma=3)
    pred_blurred = gaussian_filter(pred_mask, sigma=3)
    s_object = np.corrcoef(gt_blurred.flatten(), pred_blurred.flatten())[0, 1]
    s_object = max(0, s_object)  # Ensure non-negative correlation

    # Final S-measure
    alpha = 0.5  # Weighting factor
    s_measure = alpha * s_object + (1 - alpha) * s_region
    return s_measure


def evaluate_s_measure(model_output_dir, gt_dataset_dir, ignore_vocab_limit=True):
    """
    Evaluate S-measure across all images in DUT-OMRON dataset.
    """
    scores = []
    image_names = [f for f in os.listdir(gt_dataset_dir) if f.endswith('.png')]

    for filename in image_names:
        pred_path = os.path.join(model_output_dir, filename)
        gt_path = os.path.join(gt_dataset_dir, filename)

        if not os.path.exists(pred_path):
            print(f"Warning: Pipeline ouptut for {filename} not found. Skipping.")
            continue

        pred_mask = io.imread(pred_path, as_gray=True)
        gt_mask = io.imread(gt_path, as_gray=True)

        if (not ignore_vocab_limit) and np.all(pred_mask == 0):
            print(f"{filename} affected by YOLO vocab size. Skipping for S-measure.")
            continue

        # Ensure masks have the same size
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))

        score = s_measure(pred_mask, gt_mask)
        print(score)
        scores.append(score)

    mean_s_measure = np.mean(scores) if scores else 0
    return mean_s_measure



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
            {"role": "system",
             "content": "You are an AI that assists visually-impaired individuals by analyzing scenes."},
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
        response_format={"type": "json_object"},
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
            all_class_segments[class_name].append((d['segments'], d['confidence']))
        else:
            all_class_segments[class_name] = [(d['segments'], d['confidence'])]

    output_segment = []
    top_k = k
    for obj in ranking:
        if obj in all_class_segments:
            # output_segment.extend(all_class_segments[obj])
            # Only accept top n objects with confidence > .8
            for segment, confidence in all_class_segments[obj]:
                if confidence > .75:
                    output_segment.append(segment)
            if len(output_segment) > 0:
                top_k -= 1
                if top_k == 0:
                    break
    return output_segment


def getSalientOnlyImg(img, output_segment):
    """Grabs only the segmented objects in the top_k segments.
        Has been modified to output only the salient object segment masks for pixel-wise ground truth s-measure"""
    salient_only_img = np.zeros(img.shape[:2], dtype=np.uint8)
    for segment in output_segment:
        points_rollers = [(int(x), int(y)) for x, y in zip(segment['x'], segment['y'])]
        data_points = [tuple(point) for point in points_rollers]
        # Create a separate single-channel mask
        object_mask = np.zeros(img.shape[:2], dtype=np.uint8)  # Grayscale mask
        cv2.fillPoly(object_mask, [np.array(data_points)], 255)  # White polygon mask

        # Use the mask to extract pixels from the original image and add the extracted pixels to the final image
        print(salient_only_img.shape, object_mask.shape)
        salient_only_img = cv2.bitwise_or(salient_only_img, object_mask)

    return salient_only_img


def main(args):
    assert(len(args) > 1)
    num_workers = int(args[1])

    gt_dir = os.path.join("DUT_OMRON_dataset", "pixelwise_gt")
    orig_imgs_dir = os.path.join("DUT_OMRON_dataset", "orig_images")
    model_output_dir = os.path.join("eval_output_images")
    # existing_outputs = os.listdir(model_output_dir) # load balancing across workers
    # orig_image_fnames_todo = [f for f in os.listdir(orig_imgs_dir) if f.endswith('.jpg') and ((f[:-4]+'.png') not in existing_outputs)]
    # num_workers = min(num_workers, len(orig_image_fnames_todo))
    # workers = [Worker(i, orig_image_fnames_todo[round(i*(len(orig_image_fnames_todo)/num_workers)):round((i+1)*(len(orig_image_fnames_todo)/num_workers))]) for i in range(num_workers)]
    # for i, worker in enumerate(workers):
    #     print(f"starting worker {i}")
    #     worker.start()
    #
    # for worker in workers:
    #     worker.join() # wait for all workers to finish
    # print(f"all workers finished")
    print(f"Mean S-measure: {evaluate_s_measure(model_output_dir=model_output_dir, gt_dataset_dir=gt_dir, ignore_vocab_limit=False):.4f}") # output result

    return 0


if __name__ == '__main__':
    # Setup openai
    load_dotenv(find_dotenv())  # read local .env file
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = openai_api_key
    os.environ['OPENAI_API_KEY'] = openai_api_key

    main(sys.argv)

