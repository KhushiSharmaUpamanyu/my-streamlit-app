STREAMLIT INTERFACE LINK 
https://my-app-app-2nbtdpahjnf7xuggadfapc.streamlit.app/

# Multilingual Binary Text Classification with XLM-RoBERTa
This project implements a binary text classification model using XLM-RoBERTa, capable of handling paired text inputs and imbalanced datasets. It is designed for multilingual data, making it suitable for applications where inputs are in multiple languages.
## Overview
The model predicts a binary label (0 or 1) based on two input text fields: origin_query and category_path. It uses transformer-based embeddings from XLM-RoBERTa and fine-tunes them on the training dataset. The code also manages missing values gracefully and ensures proper tokenization, padding, and truncation.
## Key Features
### 1)Paired input handling: 
Combines information from two text columns to improve prediction.
### 2)Multilingual support: 
Leverages XLM-RoBERTa’s ability to understand multiple languages.
### 3)Class imbalance handling: 
Computes weights from training labels and uses weighted cross-entropy loss during training.
### 4)Evaluation metrics: 
Tracks both accuracy and F1 score for the positive class, ensuring balanced performance on minority classes.
### 5)GPU acceleration: 
Supports CUDA with mixed-precision training for faster execution.
## Workflow
### 1)Data Loading and Preprocessing:
Loads CSV files, splits the training data into train and validation sets, converts labels to integers, and tokenizes paired inputs.
### 2)Class Weight Calculation:
Counts the number of samples in each class to compute weights, which help the model focus on underrepresented classes.
### 3)Model Training:
Fine-tunes XLM-RoBERTa using a custom trainer that applies the weighted loss. Training progress is logged, and the best-performing model is saved automatically.
### 4)Evaluation:
Evaluates the trained model on the test set using accuracy and F1 score for the positive class.
### 5)Prediction:
Generates predictions for the test set and provides the distribution of predicted classes, allowing inspection of model bias.
## Results
The model achieves an F1 score of 0.77 on the positive class for the test set.

### The Project repo entails the 5 slide deck and all code files

