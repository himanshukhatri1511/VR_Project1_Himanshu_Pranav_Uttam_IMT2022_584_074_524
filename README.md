# VR_Project: Face Mask Detection, Classification, and Segmentation


<h3 align = "center">
Himanshu Khatri(IMT2022584)<br>
Pranav Laddhad(IMT2022074)<br>
Uttam Hamsaraj(IMT2022524)</h3>

# Introduction

Develop a computer vision solution to classify and segment face masks in images. The
project involves using handcrafted features with machine learning classifiers and deep
learning techniques to perform classification and segmentation.

# Dataset and Libraries used

A labeled dataset containing images of people with and without face masks and a Masked
Face Segmentation dataset with ground truth face masks was given.

- For the dataset corresponding to the binary classification task, we have a total of
    4,095 images, with 2,165 images labeled as with_maskand 1,930 images labeled
    as without_mask, indicating that the dataset is well-balanced.<br>
    Dataset: https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset
- For the dataset corresponding to the binary segmentation task, we have a total of
    9,382 images in the face_crop directory, along with 9,382 corresponding labels
    (segmented output masks). Each input image is an RGB image of size (128, 128,
    3), while the segmented binary mask output is a grayscale image of size (128, 128,
    1).<br>
    Dataset: https://github.com/sadjadrz/MFSD

The libraries used in the project are as follows:

- **TensorFlow & Keras**: Deep learning model building and training.
- **OpenCV & scikit-image**: Image processing and feature extraction.
- **matplotlib**: For data visualization.
- **scikit-learn**: For machine learning model implementation and evaluation.
- **NumPy**: Numerical computations.


#Binary Classification(Task A and B)
# Task A: Binary Classification Using Handcrafted Features and ML Classifiers

## Overview

This part aims to classify face images into two categories: "with mask" and "without mask" using handcrafted features and machine learning classifiers. The extracted features include shape, texture, and edge information, which are then used to train and evaluate Support Vector Machine (SVM) and Multi-Layer Perceptron (MLP) models.

## Methodology

### 1. Dataset Preparation
- Face images were loaded from the dataset and converted to grayscale.
- Images were resized to 64x64 pixels for consistency.
- Histogram equalization was applied to improve contrast.

### 2. Feature Extraction
The following handcrafted feature extraction techniques were applied:
- **Histogram of Oriented Gradients (HOG)**: Captures shape and texture information by computing gradients.
- **Local Binary Patterns (LBP)**: Encodes texture information using neighborhood pixel differences.
- **Canny Edge Detection**: Highlights significant edges in the image.
- **Sobel Edge Detection**: Computes gradients in both x and y directions to extract edge information.
- Features from all methods were concatenated into a single feature vector.

### 3. Feature Transformation
- Standardization was applied using `StandardScaler` to normalize feature values.
- Principal Component Analysis (PCA) was used to reduce dimensionality while retaining important variance.

### 4. Model Training and Evaluation
Two machine learning models were trained and evaluated using **5-fold Stratified Cross-Validation**:
- **Support Vector Machine (SVM)**: Grid search was performed to find the best hyperparameters for kernel type and regularization parameter.
- **Multi-Layer Perceptron (MLP)**: A neural network with varying hidden layer sizes (`50`, `100`, and `(50,50)`) and activation functions (`relu` and `tanh`) was trained with up to `500` iterations.

## Results

### Model Performance Comparison
| Model | Test Accuracy |
|-------|--------------|
| SVM   | 92.26%       |
| MLP   | 90.17%       |

### Classification Report
#### SVM Performance
- **Precision**: 95% (Class 0), 90% (Class 1)
- **Recall**: 89% (Class 0), 96% (Class 1)
- **Overall Accuracy**: 92.26%

#### MLP Performance
- **Precision**: 91% (Class 0), 90% (Class 1)
- **Recall**: 88% (Class 0), 92% (Class 1)
- **Overall Accuracy**: 90.17%

## Observations
- SVM outperformed MLP in classification accuracy.
- SVM showed better generalization due to its ability to effectively separate classes in a high-dimensional space.
- MLP, despite slightly lower accuracy, demonstrated strong performance in learning complex patterns.
- Adding more layers in MLP did not significantly improve accuracy, possibly due to overfitting or vanishing gradients.

## Conclusion
Both models effectively classified masked and unmasked faces, with SVM achieving higher accuracy. Future work could explore deep learning-based approaches or fine-tuning MLP hyperparameters for improved results.

# Task B : Face Mask Detection Using Convolutional Neural Networks (CNN)

## Overview
This part implements a Convolutional Neural Network (CNN) to classify face images into two categories: "with mask" and "without mask." The CNN model leverages deep learning techniques to automatically extract features, achieving higher accuracy compared to traditional machine learning approaches.

---

## Methodology

### 1. Dataset Preprocessing
- Images were converted to grayscale and resized to **64x64 pixels**.
- **Histogram equalization** was applied to enhance contrast.
- The dataset was split into **80% training** and **20% testing** sets.

### 2. CNN Architecture
A **sequential CNN model** was implemented with the following layers:
- **Three convolutional layers** (32, 64, and 128 filters).
- **Max-pooling layers** for downsampling.
- **Dropout layers** (rates: 0.3, 0.5) for regularization.
- A **dense output layer** with sigmoid activation for binary classification.

### 3. Training Configuration
- **Optimizers tested**: Adam vs. SGD.
- **Learning rates**: 0.001 vs. 0.0001.
- **Training duration**: 20 epochs with early stopping.
- **Learning rate reduction** was applied to optimize training.
- Multiple models were saved for comparison.

---

## Hyperparameter Experiments
The following combinations were tested:

| Optimizer | Dropout Rate | Learning Rate | Accuracy |
|-----------|--------------|---------------|----------|
| Adam      | 0.3          | 0.001         | **94.97%** |
| Adam      | 0.5          | 0.001         | 93.12%   |
| SGD       | 0.3          | 0.001         | 89.45%   |
| SGD       | 0.5          | 0.0001        | 85.20%   |

### Key Findings:
- **Adam** consistently outperformed **SGD**.
- **Dropout = 0.3** yielded the best accuracy (94.97%).
- Lower learning rates (e.g., 0.0001) with **SGD** led to poor performance.
- Adding more convolutional layers did **not** improve accuracy (risk of overfitting).

---

## Results

### Performance Comparison
| Model       | Accuracy | Training Time |
|-------------|----------|---------------|
| CNN (Adam)  | 94.97%   | 6-7 minutes   |
| SVM         | 92.26%   | 1-1.5 minutes |
| MLP         | 90.17%   | 1-1.5 minutes |

### Observations
1. **CNN outperformed SVM and MLP** by a significant margin.
2. **Adam optimizer** was more effective than SGD.
3. **Dropout = 0.3** provided the best balance between generalization and accuracy.
4. **Training time** for CNN was longer (~6-7 minutes) compared to traditional ML models (~1-1.5 minutes).

---

## Conclusion
- The CNN model achieved **94.97% accuracy**, surpassing traditional methods (SVM: 92.26%, MLP: 90.17%).
- Critical factors for success:
  - **Optimizer choice** (Adam > SGD).
  - **Moderate dropout** (0.3) for regularization.
  - **Learning rate tuning** (0.001 worked best).
- **Future Work**:
  - Experiment with **data augmentation** to improve robustness.
  - Test advanced architectures like **ResNet** or **MobileNet**.
  - Fine-tune deeper layers for potential gains.

---


## Task C : Region Segmentation Using Traditional Techniques

## Objective
In this part, we implement a region-based segmentation approach to identify and segment mask regions for faces classified as **"with mask."**  
The predicted segmentation masks are compared with the provided ground truth masks and evaluated using **Intersection over Union (IoU)** and **Dice Score**.

## Input Dataset
- The dataset consists of **8,225 images**, with cropped face regions extracted for individuals wearing masks.
- Each image has a corresponding **ground truth segmentation mask**.

## Methods Used

### 1. Preprocessing
- Images are converted to **HSV (Hue, Saturation, Value)** and **Grayscale** formats.
- **Gaussian Blurring** is applied to reduce noise and smooth the image for better segmentation.

### 2. Segmentation Techniques

#### a) Color-Based Segmentation (HSV Thresholding)
- **HSV thresholding** is applied to segment potential mask regions using specific color ranges.
- A **binary mask** is generated, highlighting the detected mask regions.

#### b) Threshold-Based Segmentation (Otsu’s Method)
- **Otsu’s Thresholding** is used to determine an optimal threshold value automatically.
- The output is a **binary mask**, where pixels above the threshold are set to white (255) and pixels below are set to black (0), ensuring clear separation of mask regions.

#### c) Combining Segmentation Results
- The masks from **HSV thresholding** and **Otsu’s method** are combined using a **bitwise OR operation** to improve segmentation accuracy.

### 3. Mask Refinement
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

- **Average IoU Score:** **0.35**  
- **Average Dice Score:** **0.50**  

## Output
- The **top 5 images** with the highest segmentation accuracy (highest IoU scores) are visualized.

![Figure_1](https://github.com/user-attachments/assets/a1e9f02a-fc9e-4069-bd89-67c3e633e55e)


![Figure_3](https://github.com/user-attachments/assets/56f510da-dc07-4241-9be4-9451ecc5360a)

## Observations

- The **IoU** and **Dice Score** values are relatively low when using traditional segmentation techniques.
- This is due to variations in **lighting conditions, mask colors, and background noise**, which affect threshold-based methods.
- Traditional techniques like **HSV thresholding and Otsu’s method** rely on predefined rules and are less adaptive to diverse real-world conditions.








## Task D :  Mask Segmentation Using U-Net

## Introduction 

This part of the project focuses on Mask Segmentation Using U-Net, leveraging the powerful U-Net architecture and the goal is to classify each pixel of an input image into predefined categories, with a focus on differentiating between mask and non-mask regions. 

### Key objectives include: 

- **Training a U-Net Model**: Developing and fine-tuning a U-Net model for accurate segmentation of mask regions in input images. 
- **Performance Comparison**: Evaluating the performance of the U-Net model against traditional segmentation methods using metrics such as Intersection over Union (IoU) and Dice score. 

---

## Dataset 

### Source: 
The dataset comprises input images and their corresponding binary masks (ground truth). Each mask indicates the target regions in the input images. 

### Structure: 
The dataset used in this project is the MSFD dataset, which contains two folders: 

- **face_crop**: This folder contains 9,382 cropped face images. 
- **face_crop_segmentation**: This folder contains the corresponding 9,382 binary segmentation masks. Each mask highlights the regions of interest (mask and face) in the input images using binary classification. 

---
## Methodology

### Step 1: Dataset Preparation 

#### Data Loading: 
Images and masks are loaded using TensorFlow's `load_img()` and converted to arrays using `img_to_array()`. 

#### Splitting: 
The dataset is split into training and testing sets using an 80-20 split (`train_test_split`).  
- **Training set**: 7505 samples  
- **Testing set**: 1877 samples 

---

### Step 2: Preprocessing 

The preprocessing step ensures the dataset is prepared for effective training. The following transformations are applied: 

- **Normalization**: Images are normalized to a range of [0, 1], which is crucial for stabilizing the learning process during model training. 
- **Binarization**: Masks are binarized using a threshold (> 0.5). This ensures the masks have values of either 0 or 1, suitable for binary segmentation tasks. 
- **Channel Dimensions**: Single-channel grayscale masks are expanded to include a channel dimension, ensuring compatibility with the U-Net architecture. Grayscale images are converted to RGB by repeating the single channel across all three channels. 

After preprocessing, the input images have dimensions of (128, 128, 3), representing the resized RGB images, and the corresponding output masks have dimensions of (128, 128, 1), representing the binarized single-channel masks. 

---

### Step 3: Model Architecture 

#### U-Net Model: 

A custom U-Net model is built using Keras. 

- **Input Layer**:  
   The model takes input images with dimensions (128, 128, 3). 

- **Encoder**:  
   Consists of multiple convolutional layers where features are extracted using filters.  
   - Each convolutional block includes two `Conv2D` layers with a specified number of filters, kernel size (3x3), activation function (e.g., ReLU), and `padding='same'` to preserve spatial dimensions.  
   - At each stage, the output is downsampled using a `MaxPooling2D` layer (pool size: 2x2).  
   - The encoder progressively doubles the number of filters at each level to capture finer features at deeper layers.  
   - Skip connections are stored at each stage to transfer spatial information to the decoder. 

- **Bottleneck**:  
   This stage operates at the lowest resolution of the feature map.  
   - Includes two `Conv2D` layers with the highest number of filters to extract the most abstract features. 

- **Decoder**:  
   - The decoder upsamples the feature maps back to the original input size using `Conv2DTranspose` layers (strides: 2x2, kernel size: 2x2, padding='same').  
   - Each upsampling step is followed by a concatenation with the corresponding skip connection from the encoder, enabling the model to recover spatial details.  
   - Further, two `Conv2D` layers are applied to refine the upsampled features. 

- **Output Layer**:  
   - The final layer is a `Conv2D` layer with 1 filter and kernel size (1x1).  
   - The activation function is set to `sigmoid`, producing an output mask with dimensions (128, 128, 1), where pixel values range between 0 and 1. 

#### Model Parameters: 
- **Number of Filters**: `num_filters`, defining the base number of filters in the first convolutional block. 
- **Number of Layers**: `num_layers`, representing the number of encoder-decoder blocks. 
- **Activation Function**: `activation`, such as ReLU, applied after each convolution operation. 

### Hyperparameters: 

- Number of filters: [32, 64] 
- Number of layers: [3, 4] 
- Activation function: relu 
- Batch size: [16, 32] 
- Learning rates: [1e-4, 1e-3] 

---

### Step 4: Grid Search for Hyperparameter Tuning 

#### Grid Search: 
To identify the optimal hyperparameters for the U-Net model, a grid search approach is employed. This involves: 

- Defining a **Parameter Grid** containing various combinations of hyperparameters. 
- Using `ParameterGrid` from `sklearn.model_selection` to generate all possible combinations of the defined hyperparameters. 

#### Hyperparameter Tuning Process: 

1. **Model Building**:  
   The `build_unet_model` function is used to construct the U-Net model based on the current combination of hyperparameters from the grid. 

2. **Training and Evaluation**:  
   - For each hyperparameter combination, the model is compiled, trained, and validated.  
   - Performance is assessed using validation loss and segmentation metrics, which include pixel-level metrics tailored to the segmentation task. 

#### Custom Metrics: 

Since standard Keras metrics are designed for classification tasks, custom metrics were implemented: 

- **Intersection over Union (IoU)**: Calculates the overlap between predicted and ground truth regions. 
- **Dice Score**: Measures the similarity between predicted and ground truth regions. 

These metrics are crucial for segmentation tasks as they provide pixel-wise evaluation. 

#### Best Model Selection: 

After evaluating all combinations, the hyperparameters corresponding to the model with the lowest validation loss are selected as the best. 

#### Metrics: 
Performance is evaluated using the following metrics: 

- **Validation Loss and Accuracy**: To assess the overall performance during training. 
- **Segmentation Metrics**: Include IoU, Dice score, precision, and recall, providing insights into pixel-level accuracy. 

---

### Step 5: Training the Best Model 

The best hyperparameters are used to train the final model for 6 epochs with 20% validation data. 

---

## Hyperparameters and Experiments 

### Hyperparameters Tried: 

- Number of Filters: [32, 64] 
- Number of Layers: [3, 4] 
- Learning Rates: [1e-4, 1e-3] 
- Batch Sizes: [16, 32] 
- Activation: relu 

### Results of Grid Search: 

**Best Hyperparameters**: 

- Number of Filters: 64 
- Number of Layers: 4 
- Learning Rate: 1e-4 
- Batch Size: 16 

**Best Validation Loss**: 0.1204 

---

## Results 

### Evaluation Metrics on the Test Set: 

The best model, selected based on validation loss, was evaluated on the test set with the following metrics: 

- **IoU**: 0.87 
- **Dice Score**: 0.93 
- **Precision**: 0.91 
- **Recall**: 0.90 
- **Accuracy**: 0.94 

### Visualization Results: 

Ground truth and predicted masks closely match, demonstrating effective segmentation.  
Examples of input images, ground truth masks, and predictions are visualized for qualitative analysis. 

