import numpy as np
import cv2
import rasterio
import pye57
import os
import argparse

# Constants
IMAGE_WIDTH = 4096 * 2
IMAGE_HEIGHT = 2048 * 2

def load_mask(mask_path, width, height):
    print(f"[INFO] Loading mask from: {mask_path}")
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    if mask.shape != (height, width):
        print(f"[INFO] Resizing mask from {mask.shape} to ({height}, {width})")
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.uint16)

def load_e57_points_and_origin(file_path):
    print(f"[INFO] Reading E57 file: {file_path}")
    e57 = pye57.E57(file_path)
    header = e57.get_header(0)
    sensor_pos = np.array(header.translation)

    scan = e57.read_scan(0, ignore_missing_fields=True)
    x, y, z = scan["cartesianX"], scan["cartesianY"], scan["cartesianZ"]
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    points = np.stack((x[valid], y[valid], z[valid]), axis=-1)

    print(f"[INFO] Loaded {len(points)} valid points")
    return points, sensor_pos

def spherical_projection_and_mask(points, sensor, mask, width, height):
    rel = points - sensor
    r = np.linalg.norm(rel, axis=1)
    valid = r > 0

    rel = rel[valid]
    points_valid = points[valid]
    r = r[valid]
    x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]

    theta = np.arctan2(y, x)
    theta = (theta + np.pi) / (2 * np.pi)
    phi = np.arccos(z / r) / np.pi

    px = np.clip((theta * width).astype(np.int32), 0, width - 1)
    py = np.clip((phi * height).astype(np.int32), 0, height - 1)

    labels = mask[py, px]
    return points_valid, labels

def save_ply(points, labels, path):
    print(f"[INFO] Saving PLY to: {path}")
    with open(path, "w") as f:
        f.write(f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar classification
end_header
""")
        np.savetxt(f, np.column_stack((points, labels)), fmt="%.6f %.6f %.6f %d")
    print(f"[✓] Done: {path}")

def main():
    parser = argparse.ArgumentParser(description="Project segmentation mask onto E57 point cloud and export labeled PLY.")
    parser.add_argument("-input_e57", required=True, help="Path to the input .e57 file")
    parser.add_argument("-input_mask", required=True, help="Path to the segmentation mask (TIFF)")
    args = parser.parse_args()

    input_e57 = args.input_e57
    input_mask = args.input_mask

    if not os.path.exists(input_e57):
        raise FileNotFoundError(f"Missing E57 file: {input_e57}")
    if not os.path.exists(input_mask):
        raise FileNotFoundError(f"Missing mask file: {input_mask}")

    base_name = os.path.splitext(os.path.basename(input_e57))[0]
    output_ply = f"{base_name}_labeled.ply"

    mask = load_mask(input_mask, IMAGE_WIDTH, IMAGE_HEIGHT)
    points, sensor_pos = load_e57_points_and_origin(input_e57)
    points_valid, labels = spherical_projection_and_mask(points, sensor_pos, mask, IMAGE_WIDTH, IMAGE_HEIGHT)
    save_ply(points_valid, labels, output_ply)

if __name__ == "__main__":
    main()