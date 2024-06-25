# NULLCLASS_INTERN_PROJECT__3
# Nationality, Age, Emotion, and Dress Color Prediction

This project aims to predict the nationality, age, emotion, and dress color of a person from an uploaded image using deep learning models. The models are trained on the CelebA dataset and can handle different scenarios based on the predicted nationality and age.

## Features
![image](https://github.com/ARAVINDAN20/NULLCLASS_INTERN_PROJECT__3/assets/116174602/df21fa57-f5e7-465d-ba44-d970da6a068c)
- Predict the nationality of a person from an uploaded image
- If the predicted nationality is Indian, the model also predicts age, dress color, and emotion
- If the predicted nationality is United States, the model predicts age and emotion
- If the predicted nationality is African, the model predicts emotion and dress color
- For other nationalities, the model predicts nationality and emotion
- Rejects age predictions outside the valid range (10-60 years)
- Provides a user-friendly GUI for uploading images and displaying results

## Prerequisites

- Python 3.7 or higher
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Pillow
- Tkinter

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/nationality-age-emotion-detection.git
   ```

2. Install the required dependencies:
   ```bash
   pip install tensorflow keras numpy pandas scikit-learn opencv-python pillow
   ```

3. Download the pre-trained models from the provided links and place them in the appropriate directories:
   - [Emotion Detection](https://drive.google.com/drive/folders/10khqBH1FDnEn8MYGHWxgy8XegZvnR8rz?usp=sharing)
   - [Model](https://drive.google.com/drive/folders/10khqBH1FDnEn8MYGHWxgy8XegZvnR8rz?usp=sharing)
   - [Nationality and Age Prediction Model](https://drive.google.com/drive/folders/10khqBH1FDnEn8MYGHWxgy8XegZvnR8rz?usp=sharing)

## Usage

1. Run the GUI application:
   ```bash
   python nationality_detection_GUI.py
   ```

2. Click the "Upload Image" button to select an image.
3. Click the "Predict" button to get the nationality, age, emotion, and dress color predictions.
4. The results will be displayed in the GUI.

## Models

The project utilizes the following models:

1. **Nationality and Age Prediction Model**: Trained on the CelebA dataset using the notebook provided in the [Nationality and Age Prediction Model]https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
2. **Emotion Detection Model**: Trained on the Emotion Detection dataset using the notebook provided in the [Emotion Detection Model]https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer
3. **Age and Gender Detection Model**: Trained on the utkface dataset using the notebook provided in the [Age and Gender Detection Model]https://www.kaggle.com/datasets/jangedoo/utkface-new
   
## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The CelebA dataset used for training the models
- The Keras and TensorFlow libraries for deep learning
- The OpenCV library for image processing
- The Pillow library for image manipulation
- The Tkinter library for creating the GUI


