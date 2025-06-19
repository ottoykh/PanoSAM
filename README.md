## PanoSAM: Unsupervised Segmentation Using the Meta Segment Anything Model for Point Cloud Data from Panoramic Images

![PanoSAM Overview](https://raw.githubusercontent.com/ottoykh/PanoSAM/refs/heads/main/PanoSAM.jpg)
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
TLS2Pano:
* Support for LAS point clouds, allowing for averaging based on the viewpoint to generate spherical imagery.
* Support for E57 point clouds, using the scanner placement point to create spherical images.
* Improvement of LAS point cloud functionality to enable users to define the scanner placement point.

PanoSAM:
* Support for both the Segment Anything and Segment Anything HQ models.
* Enhancement of segmentation with an instant segmentation model.

Pano2TLS:
* Currently, the Pano2TLS function only supports LAS point clouds.
* Enhancements and improvements will be made to support E57 point clouds.

### Demonstration 
![PanoSAM Demo1](https://raw.githubusercontent.com/ottoykh/PanoSAM/refs/heads/main/PanoSAM2.jpg)
![PanoSAM Demo2](https://raw.githubusercontent.com/ottoykh/PanoSAM/refs/heads/main/PanoSAM3.jpg)
