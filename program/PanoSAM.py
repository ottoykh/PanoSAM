import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline
import torch
import tifffile

def setup_device():
    device = 0 if torch.cuda.is_available() else -1
    print(f"[{timestamp()}] Step 1: Environment setup complete. Using {'GPU' if device == 0 else 'CPU'}.")
    return device

def timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def load_model(device):
    print(f"[{timestamp()}] Step 2: Loading SAM Model...")
    try:
        generator = pipeline("mask-generation", model="syscv-community/sam-hq-vit-huge", device=device)
        print(f"[{timestamp()}] > SAM Model loaded successfully.")
        return generator
    except Exception as e:
        print(f"[{timestamp()}] > Error loading SAM Model: {e}")
        exit(1)

def segment_image(image_path, generator):
    print(f"[{timestamp()}] Step 3: Starting image segmentation...")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = image.resize(original_size)
    print(f"[{timestamp()}] > Loaded image. Size: {original_size}")

    outputs = generator(image, points_per_batch=256)

    if "masks" not in outputs or not outputs["masks"]:
        raise ValueError("No masks generated.")

    masks = outputs["masks"]
    print(f"[{timestamp()}] > {len(masks)} masks generated.")

    h, w = masks[0].shape
    labeled_mask = np.zeros((h, w), dtype=np.uint16)
    for i, mask in enumerate(masks):
        labeled_mask[mask] = i + 1
    print(f"[{timestamp()}] > Labeled mask created.")

    overlay = np.zeros((h, w, 4))
    np.random.seed(42)
    for label in np.unique(labeled_mask):
        if label == 0:
            continue
        color = np.random.rand(3)
        overlay[labeled_mask == label] = np.append(color, 0.5)
    print(f"[{timestamp()}] > Overlay generated.")

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_mask_path = f"{base_name}_mask.tif"
    overlay_path = f"{base_name}_overlay.png"

    tifffile.imwrite(output_mask_path, labeled_mask)
    print(f"[{timestamp()}] > Mask saved: {output_mask_path}")

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(overlay)
    plt.title("Segmented Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(overlay_path)
    plt.close()
    print(f"[{timestamp()}] > Overlay saved: {overlay_path}")

    return output_mask_path, overlay_path

def main():
    parser = argparse.ArgumentParser(description="Segment image using SAM-HQ model.")
    parser.add_argument("-input", "--input", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    start_time = time.time()
    device = setup_device()
    generator = load_model(device)
    segment_image(args.input, generator)
    end_time = time.time()
    print(f"[{timestamp()}] Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
