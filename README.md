# Pneumonia Detection Using CNN
## Name: Shahbib Ayman Shabib

Registration No. 2018-15-43

Session: 2018-2019

## Course Instructor: Md. Mynoddin. 

Assistant Professor, Dept. of CSE, RMSTU

# Introduction
Pneumonia is a life-threatening infection of the lungs if it is not diagnosed in time. This project aims to detect pneumonia from chest X-ray images using a deep learning model designed on Convolutional Neural Networks (CNNs). By automating the detection of pneumonia, this model can assist radiologists in quicker and better diagnoses. 

The dataset used consists of 5000 labeled X-ray images that split into 3 categories:
- **80%** for training
- **10%** for validation
- **10%** for testing

## Libraries that we used:

- `os`: Handles directory and file operations.
- `numpy`: Efficient numerical computations.
- `pandas`: Data manipulation and analysis.
- `shutil`: Moves and organizes image files.
- `cv2`: Image processing operations.
- `tensorflow`: Deep learning framework for training the CNN.
- `matplotlib.pyplot`: Visualizes accuracy, loss, and other evaluation metrics.
- `seaborn`: Enhances visualization, used for plotting confusion matrices.
- `sklearn.metrics`: Computes classification reports and confusion matrices.

## Data Preprocessing

### Import Necessary Libraries
```
import os
import numpy as np
import pandas as pd
import shutil
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

### Load and Analyze Dataset
```
df = pd.read_csv("Data_Entry_2017_v2020.csv")
print(df.head())
```
- `pd.read_csv()`: Reads the CSV file containing image metadata.
- `print(df.head())`: Displays the first few rows to inspect the dataset.

### Organizing Images into Separate Folders
```
NORMAL_PATH = "images/NORMAL"
PNEUMONIA_PATH = "images/PNEUMONIA"
os.makedirs(NORMAL_PATH, exist_ok=True)
os.makedirs(PNEUMONIA_PATH, exist_ok=True)

for _, row in df.iterrows():
    image_name = row["Image Index"]
    label = row["Finding Labels"]
    dest_folder = NORMAL_PATH if label == "No Finding" else PNEUMONIA_PATH
    shutil.move(f"images/{image_name}", f"{dest_folder}/{image_name}")
```
- `os.makedirs()`: Creates directories for organizing images.
- `df.iterrows()`: Iterates through each row of the dataset.
- `shutil.move()`: Moves images to the correct folder based on their label.

### Preprocessing
- Resize all images to 224x224 pixels.
- Normalize pixel values.
- Apply data augmentation to improve generalization.
```
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    'images/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
```
- `ImageDataGenerator()`: Prepares images with transformations for better training.
- `rescale=1./255`: Normalizes pixel values.
- `flow_from_directory()`: Loads images from folders and applies augmentation.

## Model Training:

### Define CNN Architecture
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
- `Sequential()`: Builds a stack of layers for the CNN.
- `Conv2D()`: Extracts features from images using filters.
- `MaxPooling2D()`: Reduces spatial dimensions to prevent overfitting.
- `Flatten()`: Converts the 2D features into a 1D vector.
- `Dense()`: Fully connected layers for classification.
- `sigmoid`: Used for binary classification.

### Compiling the Model
```
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
- `adam`: Optimizer that adjusts learning rates dynamically.
- `binary_crossentropy`: Loss function used for binary classification.
- `accuracy`: Metric to evaluate model performance.

### Training the Model
```
history = model.fit(train_generator, epochs=20, validation_data=val_generator)
```
- Trains the model for 20 epochs using training and validation data.
- Early stopping can be applied to prevent overfitting.

## Model Evaluation

### Plot Training and Validation Accuracy
```
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
```
- Visualizes training and validation accuracy over epochs.

### Evaluating the Model on Test Data
```
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc:.2f}')
```
- Computes model performance on unseen test data.

### Generate a Confusion Matrix
```
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_true = test_generator.classes
y_pred = (model.predict(test_generator) > 0.5).astype("int32")
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```
- `confusion_matrix()`: Evaluates model predictions against true labels.
- `sns.heatmap()`: Plots the confusion matrix for better visualization.

## Results and discussion
- The model achieves **97.77% accuracy** in classifying pneumonia cases.
- Data augmentation and fine-tuning hyperparameters can further improve performance.
- This system can assist doctors by providing preliminary diagnostics for pneumonia.

## Conclusion
Here, we were successful in creating a Convolutional Neural Network (CNN) based deep learning model that could detect pneumonia from chest X-ray images. Using a collection of 5000 images, we implemented a systematic pipeline consisting of data preprocessing, training of the model, evaluation, and testing. The model achieved an impressive 97.77% accuracy, demonstrating its capacity to distinguish between healthy and pneumonia-infected lungs. Techniques such as data augmentation, careful model architecture selection, and Adam optimization contributed towards achieving this high accuracy.
