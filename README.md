# Deep Learning for Pneumonia Detection

This project focuses on the classification of Chest X-Ray images to detect Pneumonia using various Deep Learning architectures and an Ensemble method.

## Authors
**Realized by:**
* **Oussema Zghal**
* **Mohamed Amine Salhi**

## Project Overview
The goal of this project is to develop a robust model for binary classification (Normal vs. Pneumonia) using the Chest X-Ray dataset. We implemented and compared several state-of-the-art Convolutional Neural Networks (CNNs) using Transfer Learning, as well as a custom CNN baseline.

## Models Implemented
We explored the following architectures:
* **Custom CNN Base**: A baseline Convolutional Neural Network trained from scratch.
* **VGG16**: Transfer learning using the VGG16 backbone.
* **ResNet50**: Transfer learning using the ResNet50 backbone.
* **EfficientNetB0**: Transfer learning using the EfficientNetB0 backbone.
* **DenseNet121**: Transfer learning using the DenseNet121 backbone.
* **Ensemble Model**: A soft-voting ensemble combining the predictions of the trained models to maximize performance and generalization.

## Methodology
* **Preprocessing**: Images were resized and normalized. Data augmentation techniques were applied to prevent overfitting.
* **Training Strategy**: We employed **3-Fold Cross-Validation** for all models to ensure reliable performance metrics.
* **Evaluation**: Models were evaluated based on Accuracy, Precision, Recall (Sensitivity), F1-Score, and AUC-ROC.

## Repository Structure
* `Model_1_CNN_Base_Ensemble.ipynb`: Code for the custom CNN model.
* `Model_2_VGG16_Ensemble.ipynb`: Code for the VGG16 model.
* `Model_3_ResNet50_Ensemble.ipynb`: Code for the ResNet50 model.
* `Model_4_EfficientNetB0_Ensemble.ipynb`: Code for the EfficientNetB0 model.
* `Model_5_DenseNet121_Ensemble.ipynb`: Code for the DenseNet121 model.
* `Model_Comparison_Ensemble.ipynb`: Notebook containing the final comparison, ROC curves, and confusion matrices.
