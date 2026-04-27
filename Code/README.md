# Extremism Detection with Transformers

This repository contains a deep learning pipeline for classifying text messages as extremist or non-extremist. The project is built using the **PyTorch** backend and utilizes transformer architectures.

## Project Overview
The code implements an NLP workflow, including (non-exhaustive) environment configuration, data augmentation via back-translation, and model training with cross-validation.

## Environment Setup
The project is optimized for **Google Colab** with GPU acceleration.

### Requirements
The following libraries are required and installed within the notebook:
* `keras` (configured with `torch` backend)
* `transformers` (Hugging Face)
* `nlpaug` (for data augmentation)
* `gdown` (for dataset retrieval)
* `emoji` & `sacremoses`

## How to Use

### 1. Initialization
The notebook starts by setting the Keras backend to PyTorch and verifying GPU availability (e.g., Tesla T4).

### 2. Data Acquisition
The script automatically downloads the necessary data files from Google Drive:
* `train.csv`
* `test.csv`
* `sample_submission.csv`

### 3. Data Augmentation
To improve model generalization, the pipeline uses **Back-Translation**. It translates English messages to German and back to English using the `Helsinki-NLP/opus-mt` models to create semantic variations of the training set.

### 4. Training
The notebook supports training and comparing multiple models, including:
* **BERT** (`bert-base-uncased`)
* **RoBERTa** (`roberta-base`)
* **HateBERT** (`dehatebert-mono-english`)
* **BERTweet** (`bertweet-base`)

The training process includes **Stratified K-Fold** cross-validation, early stopping, and automatic saving of the best model weights to the `best_model/` directory.

### 5. Generating Submissions
Final predictions are generated for the test set and saved as CSV files in the `submissions/` folder.

## Reproducibility
A global seed (`42`) is set to ensure consistent results across different runs.