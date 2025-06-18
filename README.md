## PanoSAM: Unsupervised Segmentation Using the Meta Segment Anything Model for Point Cloud Data from Panoramic Images

![PanoSAM Overview](https://raw.githubusercontent.com/ottoykh/PanoSAM/refs/heads/main/PanoSAM.jpg)
This is a prototype package designed as a proof of concept for the SAM model on panoramic image segmentation for close-range captured point clouds, such as those obtained from terrestrial laser scanners. This package facilitates users in performing unsupervised segmentation in a faster and more effective way.

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

Pano2MMS:
* Viewpoint-based panoramic images are captured from the MMS. After using PanoSAM, the segmentation results can be projected onto the MMS point cloud.
* A new solution and method for merging segmented results from different captured frames need to be explored.

### Demonstration 
![PanoSAM Demo1](https://raw.githubusercontent.com/ottoykh/PanoSAM/refs/heads/main/PanoSAM2.jpg)
![PanoSAM Demo2](https://raw.githubusercontent.com/ottoykh/PanoSAM/refs/heads/main/PanoSAM3.jpg)
