import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline
import torch
import tifffile
import gradio as gr
import os

# Step 1: Setup
print("Step 1: Setting up the environment...")
device = 0 if torch.cuda.is_available() else -1
print(f"    > Device selected: {'GPU' if device == 0 else 'CPU'}")

# Step 2: Load SAM Model
print("Step 2: Loading SAM Model...")
generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=device)
print("    > SAM Model loaded successfully.")


def segment_image(image):
    print("Step 3: Starting image segmentation...")

    # Resize Image
    print("    > Resizing image...")
    raw_image = image.convert("RGB")
    original_size = raw_image.size
    resized_size = (original_size[0] // 4, original_size[1] // 4)
    raw_image = raw_image.resize(resized_size)
    print(f"    > Original size: {original_size}, Resized size: {resized_size}")

    # Run SAM Segmentation
    print("    > Running SAM segmentation...")
    outputs = generator(raw_image, points_per_batch=64)
    masks = outputs["masks"]
    print(f"    > {len(masks)} masks generated.")

    # Create Labeled Mask
    print("    > Creating labeled mask...")
    h, w = masks[0].shape
    labeled_mask = np.zeros((h, w), dtype=np.uint16)
    for i, mask in enumerate(masks):
        labeled_mask[mask] = i + 1
    print("    > Labeled mask created.")

    # Generate Overlay
    print("    > Generating overlay...")
    overlay = np.zeros((h, w, 4))  # RGBA
    np.random.seed(42)
    for label in np.unique(labeled_mask):
        if label == 0:
            continue
        color = np.random.rand(3)
        overlay[labeled_mask == label] = np.append(color, 0.5)
    print("    > Overlay generated.")

    # Save the labeled mask as TIFF
    output_path = "labeled_mask.tif"
    print("    > Saving labeled mask as TIFF...")
    tifffile.imwrite(output_path, labeled_mask)
    print(f"    > Mask saved to: {output_path}")

    # Plotting results
    print("Step 4: Plotting results...")
    plt.figure(figsize=(15, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Segmented Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(raw_image)
    plt.imshow(overlay)
    plt.title("Segmented Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("segmented_overlay.png")  # Save the overlay plot
    plt.close()  # Close the plot to avoid display issues
    print("    > Results plotted.")

    return output_path  # Return path to the saved mask


# Step 5: Gradio Interface
print("Step 5: Setting up Gradio interface...")
iface = gr.Interface(
    fn=segment_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.File(label="Download Mask"),
    title="Image Segmentation with SAM",
    description="Upload an image to segment it and visualize the results."
)

# Step 6: Launch the interface
print("Step 6: Launching the interface...")
iface.launch()
print("    > Interface launched successfully.")