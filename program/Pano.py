import os
import sys
import time
import argparse
import numpy as np
import cv2
import pye57
import tifffile
import rasterio
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline
from math import pi
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== Utilities ==========
def timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def setup_device():
    device = 0 if torch.cuda.is_available() else -1
    print(f"[{timestamp()}] Using {'GPU' if device == 0 else 'CPU'}")
    return device

def load_sam_model(device):
    print(f"[{timestamp()}] Loading SAM-HQ model...")
    try:
        return pipeline("mask-generation", model="syscv-community/sam-hq-vit-huge", device=device)
    except Exception as e:
        print(f"[{timestamp()}] Failed to load model: {e}")
        sys.exit(1)

# ========== Step 1: Panorama Generation ==========
def load_point_cloud(filename):
    e57 = pye57.E57(filename)
    data = e57.read_scan(0, intensity=True, colors=True, ignore_missing_fields=True)
    sensor_position = np.array(e57.get_header(0).translation)
    return data, sensor_position

def spherical_projection(points, distances, height, width):
    x, y, z = points
    theta = np.arctan2(y, x)
    phi = np.arccos(z / distances)
    pixel_x = ((theta + pi) / (2 * pi)) * width
    pixel_y = (phi / pi) * height
    return pixel_x.astype(np.int32), pixel_y.astype(np.int32)

def create_panorama_image(data, sensor_pos, height):
    xyz = np.stack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]])
    rel_xyz = xyz - sensor_pos[:, None]
    distances = np.linalg.norm(rel_xyz, axis=0)
    valid = distances > 0
    rel_xyz = rel_xyz[:, valid]
    distances = distances[valid]
    colors = np.stack([
        data["colorRed"][valid],
        data["colorGreen"][valid],
        data["colorBlue"][valid]
    ], axis=-1).astype(np.uint8)
    width = height * 2
    px, py = spherical_projection(rel_xyz, distances, height, width)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), np.inf, dtype=np.float32)

    def process_range(start, end):
        for i in range(start, end):
            x, y = px[i], py[i]
            if 0 <= x < width and 0 <= y < height:
                if distances[i] < depth[y, x]:
                    depth[y, x] = distances[i]
                    image[y, x] = colors[i]

    chunk_size = len(px) // os.cpu_count()
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_range, i, min(i + chunk_size, len(px))) for i in range(0, len(px), chunk_size)]
        for future in as_completed(futures):
            pass

    return image, width, height

# ========== Step 2: Segmentation ==========
def segment_image(image_path, generator):
    image = Image.open(image_path).convert("RGB")
    result = generator(image, points_per_batch=256)
    if "masks" not in result or not result["masks"]:
        raise ValueError("No masks generated.")
    masks = result["masks"]
    h, w = masks[0].shape
    labeled = np.zeros((h, w), dtype=np.uint16)
    for i, mask in enumerate(masks):
        labeled[mask] = i + 1
    mask_path = image_path.replace(".jpg", "_mask.tif")
    tifffile.imwrite(mask_path, labeled)
    return labeled, mask_path

# ========== Step 3: Label Projection ==========
def spherical_projection_mask(points, colors, sensor, mask, width, height):
    rel = points - sensor
    r = np.linalg.norm(rel, axis=1)
    valid = r > 0
    rel = rel[valid]
    r = r[valid]
    x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]
    theta = (np.arctan2(y, x) + pi) / (2 * pi)
    phi = np.arccos(z / r) / pi
    px = np.clip((theta * width).astype(np.int32), 0, width - 1)
    py = np.clip((phi * height).astype(np.int32), 0, height - 1)
    labels = mask[py, px]
    return points[valid], colors[valid], labels

def save_ply(points, colors, labels, output_path):
    with open(output_path, "w") as f:
        f.write(f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar classification
end_header
""")
        data = np.column_stack((points, colors, labels))
        np.savetxt(f, data, fmt="%.6f %.6f %.6f %d %d %d %d")
    print(f"[✓] Saved labeled PLY: {output_path}")

# ========== Main Pipeline ==========
def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="E57 Pipeline: pano -> segment -> label point cloud")
    parser.add_argument("-input_e57", required=True, help="Input .e57 file")
    parser.add_argument("--image_height", type=int, default=2048, help="Panorama image height")
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.input_e57))[0]
    pano_path = f"{base_name}_pano.jpg"
    ply_path = f"{base_name}_labeled.ply"

    print(f"[{timestamp()}] Step 1: Generating panorama")
    data, sensor_pos = load_point_cloud(args.input_e57)
    pano_img, width, height = create_panorama_image(data, sensor_pos, args.image_height)
    cv2.imwrite(pano_path, pano_img)
    print(f"[✓] Saved panorama: {pano_path}")

    print(f"[{timestamp()}] Step 2: Segmenting image")
    device = setup_device()
    generator = load_sam_model(device)
    labeled_mask, mask_path = segment_image(pano_path, generator)
    print(f"[✓] Segmentation saved: {mask_path}")

    print(f"[{timestamp()}] Step 3: Projecting labels to 3D")
    points = np.stack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]], axis=-1)
    colors = np.stack([data["colorRed"], data["colorGreen"], data["colorBlue"]], axis=-1)
    valid = ~(np.isnan(points).any(axis=1))
    points = points[valid]
    colors = colors[valid]
    labeled_points, labeled_colors, labels = spherical_projection_mask(points, colors, sensor_pos, labeled_mask, width, height)
    save_ply(labeled_points, labeled_colors, labels, ply_path)

    end_time = time.time()
    print(f"[{timestamp()}] Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()