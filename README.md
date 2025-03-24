# PART C - Region Segmentation Using Traditional Techniques

## Objective
In this part, we implement region-based segmentation approach to identify and segment mask regions for faces classified as "with mask". The predicted segmentation masks are then compared with the provided ground truth masks and evaluated using **Intersection over Union (IoU)** and **Dice Score**.

## Input Dataset
- The dataset consists of 8,225 images, with cropped face regions extracted for individuals wearing masks. These cropped images are used for segmentation.
- Each image has a corresponding **ground truth segmentation mask**.

## Methods Used

### 1. Preprocessing
- Images are loaded and converted to **HSV (Hue, Saturation, Value)** and **Grayscale** formats.
- **Gaussian Blurring** is applied to reduce noise and smooth the image for better segmentation.

### 2. Segmentation Techniques
#### a) Color-Based Segmentation (HSV Thresholding)
- **HSV thresholding** is applied to segment potential mask regions using  color ranges.
- A **binary mask** is generated, highlighting the detected mask regions.

#### b) Threshold-Based Segmentation (Otsu’s Method)
- **Otsu’s Thresholding** is then used to determine an optimal threshold value automatically. It analyzes the image histogram and finds a threshold that minimizes the variance between the two segmented regions (foreground and background).
- The output is a **binary mask**, where pixels above the threshold are set to white (255) and pixels below are set to black (0), ensuring clear separation of mask regions.

#### c) Combining Segmentation Results
- The masks from **HSV thresholding** and **Otsu’s method** are combined using a **bitwise OR operation** to improve segmentation accuracy.

### 3.Mask Refinement
- **Morphological Closing** is applied to fill small gaps in the segmented mask.
- **Morphological Opening** is used to remove noise and refine the mask boundaries.

### 4. Contour Detection and Final Mask Extraction
- **Contours** are detected from the processed mask.
- **Small contours** are filtered out to eliminate noise.
- The **largest valid contour** is selected and used as the final predicted mask.



## Evaluation

The predicted segmentation masks are evaluated against the ground truth masks using the following metrics:

- **Intersection over Union (IoU):** Measures the overlap between the predicted and ground truth masks.
- **Dice Score:** Measures the similarity between the predicted and ground truth masks.

For each image, both **IoU** and **Dice Score** are computed to assess segmentation accuracy.

-  **Average IoU Score:** **0.35** 
-  **Average Dice Score:** **0.50** 
  
## Output
- The final segmentation masks are obtained and compared with **ground truth segmentation masks** for evaluation .
- The **top 5 images** with the highest segmentation accuracy (highest IoU scores) are visualized.

 ![Figure_1](https://github.com/user-attachments/assets/a1e9f02a-fc9e-4069-bd89-67c3e633e55e)

     

