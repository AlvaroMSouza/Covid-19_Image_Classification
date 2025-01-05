# COVID-19 Diagnosis Using Radiological Imaging

## Overview

This project focuses on the diagnosis of COVID-19 through radiological imaging, specifically chest X-ray and computed tomography (CT) images. The **goal** of this work is to explore and compare the performance of a custom-built Convolutional Neural Network (CNN) with a pre-trained ResNet-50 model in classifying COVID-19 cases. The dataset consists of labeled images of chest X-rays, and the project demonstrates my ability to work with unstructured data, implement neural networks, and evaluate model performance using key metrics.

## Dataset Description

The dataset is composed of chest X-ray images, divided into training and testing sets. It is structured as follows:
- Train Dataset: Used for training the models.
- Test Dataset: Used for evaluating model performance.

## Key Characteristics:
* COVID-19 is confirmed through PCR, but radiological imaging has been identified as an effective complementary tool.
* Chest X-ray and CT images exhibit abnormalities such as bilateral ground-glass opacity and subsegmental areas of consolidation.
* Accurate and timely detection of COVID-19 is crucial to reduce false negatives and prevent unnecessary exposure to quarantine in false positive cases.

## Methodology

### 1. Data Preprocessing

Applied transformations to augment the training dataset:
 - Random horizontal flips.
 - Automatic contrast adjustment.
 - Resized all images to 128x128 pixels.
 Normalized and converted images to tensors for compatibility with PyTorch.

### 2. Model Architectures
**Custom CNN**:
- Feature extraction:
 - Two convolutional layers with ELU activation and max pooling.
 - Flattened output for classification.
- Classification head:
 - Fully connected layer for binary classification.

**ResNet-50**:
- Loaded a pre-trained ResNet-50 model.
- Replaced the fully connected layer to match the binary classification task.
- Fine-tuned the model on the COVID-19 dataset.

### 3. Training

- Optimized using the Adam optimizer with a learning rate of 0.001.
- Cross-entropy loss function used for both models.
- Trained for 20 epochs with a batch size of 32.

### 4. Evaluation
Metrics used:
 - Accuracy
 - Precision (macro-averaged)
 - Recall (macro-averaged)
 - Evaluated both models on the test dataset.
 
 Visualized results using Seaborn to compare metrics.

## Results

Comparison:
- The ResNet-50 model outperformed the custom CNN in all metrics, showcasing the power of transfer learning.
- The CNN model achieved reasonable performance, demonstrating the effectiveness of building custom architectures for smaller datasets.
- Visualization: A bar plot was created to compare the metrics of both models, emphasizing the differences in performance.

![image](https://github.com/user-attachments/assets/ae0fedbd-8f36-47b8-ba3d-20631c844d1b)

## How to Run the Project

Prerequisites:
- Python 3.8+
- Libraries:
 - PyTorch
 - TorchMetrics
 - TorchVision
 - Matplotlib
 - Seaborn
 - Pandas
 - NumPy

## Steps

1. Clone the repository: git clone https://github.com/AlvaroMSouza/Covid-19_Image_Classification

2. Navigate to the project directory: cd Covid-19_Image_Classification

3. Install the required libraries: pip install -r requirements.txt

4. Train the models by running the script:python train_and_evaluate.py

5. View results in the console and the generated visualization plot.

## Lessons Learned

- Implementing custom CNN architectures provides hands-on experience in designing neural networks.
- Transfer learning with pre-trained models like ResNet-50 can significantly boost performance, especially with limited data.
- Metrics such as precision and recall are crucial for applications like COVID-19 diagnosis, where false negatives and positives have significant consequences.

## Acknowledgments

Dataset: Chest X-ray images provided by open-source repositories.
References:
- Ng, 2020
- Huang, 2020
- Fang, 2020
- Ai, 2020

## Contact
For any questions or collaboration opportunities, feel free to contact me at alvaromotasouza@gmail.com.

