# CNN Model Comparison and Optimization for Image Classification

## Overview

This project focuses on the development, optimization, and comparison of Convolutional Neural Network (CNN) models for image classification tasks. 
It includes three custom CNN architectures (CNNModel, CNNModel2, and OptimizedCNNModel) and evaluates pretrained models such as ResNet50, MobileNetV2, and EfficientNetB0. The models are trained and tested with data augmentation techniques to enhance performance, and their effectiveness is measured using metrics like accuracy, precision, recall, F1 score, and AUC from ROC curves. The project demonstrates the impact of architectural changes, batch normalization, dropout, and hyperparameter tuning on model performance.

## Features


- Custom CNN Architectures:





- CNNModel: A CNN with four convolutional layers, batch normalization, and dropout (total parameters: ~5.26M).



- CNNModel2: A deeper CNN with four convolutional layers and max-pooling, without batch normalization (total parameters: ~5.26M).



- OptimizedCNNModel: A lightweight CNN with fewer filters per layer, reducing the parameter count (total parameters: ~1.33M).



- Pretrained Models: Evaluation of ResNet50, MobileNetV2, and EfficientNetB0, achieving up to 100% validation accuracy.



- Data Augmentation: Techniques like random horizontal flips, rotations, affine transformations, and normalization (ImageNet standards) to improve model generalization.



- Performance Metrics: Comprehensive evaluation using:





- Training and validation loss/accuracy.



- Precision, recall, and F1 score.



- ROC curves and AUC scores (0.98–0.99 across all classes for custom models).



- Training Details:





- Custom models trained for 30–40 epochs with batch size 32 and learning rate 0.001.



- Pretrained models fine-tuned for 10 epochs, achieving high accuracy (98.18%–100%).



- Visualization: Includes confusion matrices and heatmaps to analyze classification performance.



- Hyperparameter Tuning: Experiments with unfreezing convolutional layers and adjusting fully connected layers to optimize performance.
