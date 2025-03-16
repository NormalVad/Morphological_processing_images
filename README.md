# EE569 Homework Assignment
## Date: March 16, 2025
## Name: Ayush Goyal
## ID: 7184517074
## Email: ayushgoy@usc.edu

--------------------------------------------------

## Problem 1: Geometric Image Modification
### Image Warping
**M-file name:** Prob1.m
**Usage:** image_warping()
**Input images:**
- P1P2/Panda.raw
- P1P2/Cat.raw
**Output images:** 
- Panda_warped.raw
- Panda_recovered.raw
- Cat_warped.raw
- Cat_recovered.raw
**Parameters:**
- Image dimensions: 800x800x3
- Arc thickness: 128 pixels
- Circle radius: Image width (800)

**Implementation Details:**
This problem implements geometric image warping using a non-linear transformation. The code applies a special warping function to distort images and then recovers them. The transformation preserves the main diagonal and part of the second diagonal while warping the rest of the image based on boundary curves.

**Functions:**
- `warp_transform`: Applies warping to input image with red concave arcs and green arc boundaries
- `reverse_warp_transform`: Recovers the original image from warped version
- `sample_bilinear`: Performs bilinear interpolation for smooth pixel sampling
- `smoothstep`: Implements cubic smoothstep function for blending

```matlab
% To run this code:
disp('Running Problem 1: Geometric Image Modification');
figure(1);
image_warping();
disp('Done, output images are "Panda_warped.raw", "Panda_recovered.raw", "Cat_warped.raw", and "Cat_recovered.raw"');
```

--------------------------------------------------

## Problem 2: Homographic Transformation and Image Stitching
### Panorama Creation
**M-file name:** Prob2.m
**Usage:** Run the script directly
**Input images:**
- P1P2/Street_Left.raw
- P1P2/Street_Middle.raw
- P1P2/Street_Right.raw
**Output image:**
- Stitched panorama (displayed in figure 9)
**Parameters:**
- Image dimensions: 640x480x3
- Feature threshold: 0.5
- Ratio threshold: 0.75
- Distance threshold: 2.0
- SURF parameters: MetricThreshold=800, NumOctaves=4, NumScaleLevels=6

**Implementation Details:**
This code implements panoramic image stitching by detecting features in multiple images, matching corresponding points, computing homography matrices, and blending the transformed images. The implementation uses SURF features for robust point matching and homography estimation for perspective correction.

**Key Functions:**
- `transform_points` / `transform_point`: Apply homography transformation to coordinates
- `select_points`: Intelligent selection of control points from matched features
- `compute_homography`: Computes homography matrix from point correspondences
- `normalize_points`: Normalizes point coordinates for numerical stability

```matlab
% To run this code:
disp('Running Problem 2: Homographic Transformation and Image Stitching');
disp('Detecting and matching features...');
% Run the script directly
disp('Done, panorama is displayed in Figure 9');
```

--------------------------------------------------

## Problem 3: Morphological Image Processing
### Thinning and Defect Detection
**M-file name:** Prob3.m
**Usage:** Prob3()
**Input images:** 
- P3/Spring.raw (512x512)
- P3/Flower.raw (512x512)
- P3/Circle.raw (512x512)
- P3/Tree.raw (512x512)
- P3/Chest_cavity.raw (410x305)
**Output images:**
- Displayed thinned versions of input images
- Displayed corrected chest image with defects removed
**Parameters:**
- Max thinning iterations: 20
- Defect size threshold: 50 pixels

**Implementation Details:**
This problem has two parts:
1. **Part A - Thinning**: Implements a morphological thinning algorithm to reduce binary shapes to their skeleton form. The code iteratively removes boundary pixels while preserving connectivity.
2. **Part B - Defect Detection**: Detects and removes small defects (holes) in a chest cavity image using connected component labeling and size-based filtering.

**Key Functions:**
- `thin`: Implements thinning algorithm
- `labelCC`: Labels connected components in binary image
- `findDefects`: Identifies defects based on component size
- `fixDefects`: Removes detected defects from the image

```matlab
% To run this code:
disp('Running Problem 3: Morphological Image Processing');
figure(1);
Prob3();
disp('Done, results displayed in figures');
```
