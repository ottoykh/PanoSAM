import argparse
import numpy as np
import cv2
import pye57
from math import pi
import os
import sys

# ==== Functions ====

def load_point_cloud(filename):
    """
    Load an E57 point cloud file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    print(f"Loading E57 file: {filename}")
    e57 = pye57.E57(filename)
    data = e57.read_scan(0, intensity=True, colors=True, ignore_missing_fields=True)
    header = e57.get_header(0)
    sensor_position = np.array(header.translation)
    return data, sensor_position


def spherical_projection(points, distances, height, width):
    """
    Projects 3D points (Nx3) onto a 2D spherical panorama image plane.

    Returns:
        pixel_x, pixel_y: pixel coordinates for each point
    """
    x, y, z = points
    theta = np.arctan2(y, x)  # [-pi, pi]
    phi = np.arccos(z / distances)  # [0, pi]

    # Normalize theta to [0, 1] range, then scale to image width
    pixel_x = ((theta + pi) / (2 * pi)) * width
    # Normalize phi to [0, 1] range, then scale to image height
    pixel_y = (phi / pi) * height

    return pixel_x.astype(np.int32), pixel_y.astype(np.int32)

def create_pano_image(data, sensor_pos, image_height, scale_colors=False, use_sd_scaling=False):
    """
    Generate a 2D panorama image from a 3D E57 point cloud dataset.

    Parameters:
        data (dict): The point cloud data.
        sensor_pos (np.ndarray): The sensor's position.
        image_height (int): Height of the output panorama image.
        scale_colors (bool): Whether to scale colors to a 256 range.
        use_sd_scaling (bool): Whether to scale colors based on standard deviation.

    Returns:
        np.ndarray: The generated panorama image.
    """
    # Extract xyz points and compute relative coordinates to sensor
    xyz = np.stack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]])
    relative_xyz = xyz - sensor_pos[:, None]

    # Compute distances and filter valid points (distance > 0)
    distances = np.linalg.norm(relative_xyz, axis=0)
    valid_mask = distances > 0
    if not np.any(valid_mask):
        raise RuntimeError("No valid points found in the scan.")

    relative_xyz = relative_xyz[:, valid_mask]
    distances = distances[valid_mask]

    # Extract colors
    colors = np.stack([
        data["colorRed"][valid_mask],
        data["colorGreen"][valid_mask],
        data["colorBlue"][valid_mask]
    ], axis=-1).astype(np.float32)

    # Apply color scaling
    if scale_colors or use_sd_scaling:
        if use_sd_scaling:
            # Normalize colors by standard deviation
            mean = np.mean(colors, axis=0)
            std = np.std(colors, axis=0)
            colors = (colors - mean) / (std + 1e-5)  # Avoid division by zero
            colors = np.clip((colors - colors.min()) / (colors.max() - colors.min()) * 255, 0, 255)
        else:
            # Scale colors to the 256 range
            colors = np.clip(colors * 256, 0, 255)

    colors = colors.astype(np.uint8)

    # Prepare panorama image dimensions
    height = image_height
    width = image_height * 2  # Typical 2:1 equirectangular aspect ratio

    # Project 3D points to 2D spherical coordinates (pixel locations)
    pixel_x, pixel_y = spherical_projection(relative_xyz, distances, height, width)

    # Initialize empty image and depth buffer (for z-buffering)
    pano_image = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)

    # Draw points using depth buffering (only closest points retained)
    for i in range(len(distances)):
        x, y = pixel_x[i], pixel_y[i]
        if 0 <= x < width and 0 <= y < height:
            if distances[i] < depth_buffer[y, x]:
                depth_buffer[y, x] = distances[i]
                pano_image[y, x] = colors[i]

    return pano_image

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate spherical panorama images from E57 point cloud files.")
    parser.add_argument("input_file", type=str, help="Path to the input .e57 file.")
    parser.add_argument("output_file", type=str, help="Path to save the generated panorama image (e.g., output.jpg).")
    parser.add_argument("--image_height", type=int, default=4096, help="Height of the output image (default: 4096).")
    parser.add_argument("--scale_colors", action="store_true", help="Scale colors to 256 range if enabled.")
    parser.add_argument("--full_res", action="store_true", help="Reserved for future use.")
    return parser.parse_args()


# ==== Main ====

def main():
    # Parse command-line arguments
    args = parse_args()

    try:
        # Load E57 file
        data, sensor_pos = load_point_cloud(args.input_file)

        # Generate panorama image
        print("Generating panorama image...")
        pano_img = create_pano_image(
            data,
            sensor_pos,
            args.image_height,
            scale_colors=args.scale_colors
        )

        # Save the output image
        print(f"Saving output to {args.output_file}...")
        cv2.imwrite(args.output_file, pano_img)
        print("Processing complete.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()