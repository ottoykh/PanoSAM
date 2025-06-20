## PanoSAM: Unsupervised Segmentation Using the Meta Segment Anything Model for Point Cloud Data from Panoramic Images

![PanoSAM Overview](https://raw.githubusercontent.com/ottoykh/PanoSAM/refs/heads/main/img/PanoSAM.jpg)
This is a prototype package designed as a proof of concept for the SAM model on panoramic image segmentation for close-range captured point clouds, such as those obtained from terrestrial laser scanners. This package facilitates users in performing unsupervised segmentation in a faster and more effective way.

### How to use (for E57 TLS data)
1. To run the program, download the program file:
```bash
  cd program 
```
2. Project the TLS point cloud data into Panoramic image
```bash
  python e57Pano.py "e57_file.e57" "output.jpg"
```
3. Use Segment Anything HQ Model to segment the Panormaic image from TLS Point cloud data
```bash
  python PanoSAM.py -input "output.jpg"
```
4. Reproject the segmented Panormaic image back to the TLS Point cloud data
```bash
  python Pano2e57.py -input_e57 "e57_file.e57" -input_mask "output_mask.tif"
```

### Funtions
#### e57Pano processing steps:
1. Load an E57 point cloud file
2. Read scan data and extract the sensor position
3. Extract x, y, z from points

Calculate angles (θ and ϕ) from points 
θ=arctan2(y/x)(azimuth angle)
ϕ=arccos(z/distances)

4. Normalize angles to pixel coordinates

Pixel X = pixel_x=(θ+π/ 2π)⋅width
Pixel Y = pixel_y=(ϕ/π)⋅height

5. Generate a 2D panorama image from a 3D point cloud dataset
6. Extract XYZ coordinates and compute relative coordinates

relative_xyz=xyz−sensor_pos

7. Calculate distances
8. Standard Deviation Scaling of color 

colors=(colors−mean)/std+1e−5

9. Project 3D points to 2D pixel coordinates using spherical_projection
10. Draw points onto the panorama image using a depth buffer

#### PanoSAM processing steps:
1. Load the SAM (Segment Anything Model) using the transformers pipeline
2. Open the image and convert it to RGB format
3. Resize the image (though it keeps its original size)
4. Use the generator to obtain masks for the image
5. Iterate through generated masks and label them in the mask array
6. Create an overlay with random colors for each labeled segment
7. Save the labeled mask as a TIFF file using tifffile
8. Create a figure with two subplots: the original image and the segmented overlay

#### Pano2e57 processing steps:
1. Load a segmentation mask from a TIFF file
2. Load point cloud data from an E57 file and retrieve the sensor's position
3. Project 3D points onto a 2D mask based on the sensor position

Calculate the relative positions of points to the sensor

Compute the radius r for each point and filter points

Calculate spherical coordinatesθ=arctan2(y/x)(azimuth angle)ϕ=arccos(z/r)

4. Normalize and convert spherical coordinates to pixel coordinates
5. Retrieve the corresponding labels from the mask using the pixel coordinates
6. Save the projected points and their labels to a PLY file


### Demonstration 
![PanoSAM Demo1](https://raw.githubusercontent.com/ottoykh/PanoSAM/refs/heads/main/img/PanoSAM2.jpg)
![PanoSAM Demo2](https://raw.githubusercontent.com/ottoykh/PanoSAM/refs/heads/main/img/PanoSAM3.jpg)