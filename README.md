# PneumoDetect

PneumoDetect is a deep learning system for detecting pneumonia from chest X-ray images using a Convolutional Neural Network (CNN). It classifies chest X-rays as **Normal** or **Pneumonia** and provides prediction confidence.

## Live Demo

[Launch PneumoDetect](https://pneumodetect-2005.streamlit.app/)

## Features

* Detects pneumonia from chest X-ray images
* Classifies images as Normal or Pneumonia
* Provides prediction confidence
* Uses a pretrained ResNet18 architecture with transfer learning
* Streamlit-based interactive web interface
* Includes scripts for training, evaluation, and prediction

## Tech Stack

* Python
* PyTorch
* Torchvision
* ResNet18
* Streamlit
* OpenCV
* NumPy
* Pillow
* Scikit-learn

## Model

The system uses **ResNet18** with transfer learning. The final classification layer is modified for binary classification:

* Normal
* Pneumonia

Input X-ray images are resized to **224 × 224** and normalized using ImageNet normalization values before being passed to the model.

## Dataset

Dataset: Kaggle Chest X-ray Pneumonia dataset

The dataset contains:

* `train/`
* `val/`
* `test/`

with the following classes:

* `NORMAL`
* `PNEUMONIA`

The dataset is not included in this repository and must be downloaded separately.

## Installation

```bash
git clone https://github.com/pari-dudeja2005/PneumoDetect.git
cd PneumoDetect

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Run the Streamlit Application

```bash
streamlit run app.py
```

The application can also be accessed through the deployed version:

**https://pneumodetect-2005.streamlit.app/**

## Project Structure

```text
PneumoDetect/
│
├── models/
│   └── resnet18_pneumonia.pth
│
├── app.py
├── dataloader.py
├── evaluate.py
├── predict.py
├── train_model.py
├── test_env.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Evaluation

The project includes evaluation scripts for measuring:

* Accuracy
* Precision
* Recall
* F1-score

## Disclaimer

PneumoDetect is an educational AI project and is **not intended to replace professional medical diagnosis or clinical decision-making**.





