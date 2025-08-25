# Chest X-ray Classification 🩺

Welcome to the **Chest X-ray** repository! This project uses a Convolutional Neural Network (CNN) built with Keras/TensorFlow to classify chest X-ray images as either **NORMAL** ✅ or **PNEUMONIA** 🦠. The dataset consists of images from the `chest_xray` directory, divided into train, validation, and test sets.

## Project Overview 📊
- **Goal**: Detect pneumonia from chest X-rays using deep learning.
- **Dataset**: Contains folders for `NORMAL` and `PNEUMONIA` images (e.g., bacteria/virus labeled). Training data has imbalanced classes, with more pneumonia samples.
- **Model Architecture** 🛠️:
  - Convolutional layers (Conv2D with ReLU activation) for feature extraction.
  - MaxPooling2D for downsampling.
  - Flatten and Dense layers for classification.
  - Output: Binary classification (sigmoid activation).
- **Training**:
  - Optimizer: Adam 🚀
  - Loss: Binary Crossentropy 📉
  - Metrics: Accuracy 📈
  - Epochs: 10 (with validation)
- **Evaluation**: Includes loss/accuracy plots using Matplotlib. Model tested on sample images from the test set.
- **Libraries Used**: NumPy, Pandas, Seaborn, Matplotlib, OpenCV, TensorFlow/Keras 📚

## Installation ⚙️
1. Clone the repository:
   ```bash
   git clone https://github.com/SamvelStepanyan4/Chest_X-ray.git
   cd Chest_X-ray
   ```
2. Install required dependencies (use a virtual environment for best practices):
   ```bash
   pip install numpy pandas seaborn matplotlib opencv-python tensorflow
   ```
3. Download the dataset (e.g., from [Kaggle's Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)) and place it in a `chest_xray/` folder with subfolders: `train/`, `val/`, `test/`, each containing `NORMAL/` and `PNEUMONIA/`.

## Usage 🚀
1. Open the Jupyter notebook:
   ```bash
   jupyter notebook "Chest X-ray.ipynb"
   ```
2. Run the cells step-by-step:
   - Import libraries and load data.
   - Build and compile the model.
   - Train the model on the training data.
   - Evaluate and plot results.
   - Test on a sample image (e.g., from `chest_xray/test/NORMAL/`).

### Example Prediction 🔍
```python
import matplotlib.image as mpimg
import cv2
import numpy as np

# Load and preprocess a test image
test_image = mpimg.imread('chest_xray/test/NORMAL/NORMAL-3065672-0001.jpeg')
if len(test_image.shape) == 3:
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image = cv2.resize(test_image, (224, 224))
test_image = test_image.reshape(1, 224, 224, 1)

# Predict
y_pred = model.predict(test_image)
labels = ['NORMAL', 'PNEUMONIA']
result = labels[np.argmax(y_pred)]
print(f"Prediction: {result} 🩻")
```

## Results 📈
- Training Loss vs. Validation Loss plot shows convergence.
- Accuracy improves over epochs.
- Sample test: The model correctly classifies normal X-rays (e.g., `NORMAL-3065672-0001.jpeg` as `NORMAL`).

## Dataset Details 🗂️
- **Train**: ~5000+ images (more `PNEUMONIA` than `NORMAL`).
- **Test**: ~200+ images.
- Images are grayscale, resized to 224x224.

## Future Improvements 🔮
- Handle class imbalance with techniques like oversampling or class weights ⚖️.
- Add data augmentation (e.g., rotation, flip) using `ImageDataGenerator`.
- Experiment with transfer learning (e.g., VGG16 or ResNet) for better accuracy.
- Deploy as a web app using Streamlit or Flask 🌐.

## Contributing 🤝
Feel free to fork this repo, make improvements, and submit a pull request! Issues and suggestions are welcome.

## License 📄
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.