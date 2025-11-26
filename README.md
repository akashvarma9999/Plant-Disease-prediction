# 🌿 Plant Disease Prediction using CNN

A deep learning–powered system that detects plant diseases from leaf images. This project uses a Convolutional Neural Network (CNN) trained on a large Kaggle dataset to accurately classify healthy and diseased leaves across multiple plant species.

---

## 🚀 Overview

Early detection of plant diseases helps farmers reduce losses and increase crop yield.
This project builds a **95%+ accuracy** deep learning model using **TensorFlow & Keras** to classify plant leaf images into multiple disease categories.

---

## 📌 Features

* 🔬 **CNN model** trained on 50,000+ images
* 🌱 Detects multiple plant diseases
* 📊 Includes training graphs (accuracy & loss)
* 📂 Ready-to-use dataset pipeline (train/valid/test split)
* 🧪 Predict from single image (via `main.py`)
* 🖼️ Works with real leaf images uploaded by the user

---

## 📁 Project Structure

```
PlantDisease-main/
│── Train_plant_disease.ipynb     # Notebook for training the CNN model
│── Test_plant_disease.ipynb      # Notebook for model testing + visualization
│── main.py                       # Run prediction on new leaf images
│── training_hist.json            # Saved training metrics
│── Details/                      # Images & resources
│── venv/                         # Virtual environment (optional)
└── dataset/                      # Extracted Kaggle dataset (train/valid/test)
```

---

## 🗃️ Dataset

**Kaggle Dataset:**
🌐 [https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

The dataset must be extracted as:

```
dataset/
├── train/
├── valid/
└── test/
```

Each folder contains subfolders for each plant disease class.

---

## 🧠 Model Architecture

The CNN includes:

* 3 Convolution layers
* MaxPooling layers
* Dropout to reduce overfitting
* Fully connected Dense layers
* Softmax classification output

**Optimizer:** Adam
**Loss Function:** Categorical Crossentropy

---

## ⚙️ Installation

### 1️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
```

### 2️⃣ Activate Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install tensorflow keras numpy matplotlib opencv-python
```

---

## ▶️ Training the Model

Open the notebook:

```
Train_plant_disease.ipynb
```

Run all cells — the model will:

* Load dataset
* Augment images
* Train CNN
* Save model + training history

---

## ▶️ Testing the Model

Open:

```
Test_plant_disease.ipynb
```

It will:

* Load the trained model
* Evaluate accuracy on test set
* Visualize accuracy/loss curves
* Show predictions

---

## 🖼️ Predict a Single Image

Run:

```bash
python main.py --image path/to/leaf.jpg
```

The model will output:

```
Predicted Class: Tomato___Late_blight
Confidence: 97.4%
```

---

## 📊 Results

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | ~95%  |
| Validation Accuracy | ~93%  |
| Test Accuracy       | ~94%  |

Performance may vary depending on augmentation & batch size.

---

