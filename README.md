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

1. **Nationality and Age Prediction Model**: Trained on the CelebA dataset using the notebook provided in the [Nationality and Age Prediction Model]
2. **Emotion Detection Model**: Trained on the CelebA dataset using the notebook provided in the [Emotion Detection Model]https://drive.google.com/drive/folders/10khqBH1FDnEn8MYGHWxgy8XegZvnR8rz?usp=sharing
3. **Age and Gender Detection Model**: Not provided in the search results, but you can train this model using the CelebA dataset as well.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The CelebA dataset used for training the models
- The Keras and TensorFlow libraries for deep learning
- The OpenCV library for image processing
- The Pillow library for image manipulation
- The Tkinter library for creating the GUI


Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/21664895/5f794f27-2022-4f48-b535-619599af1aee/emotion-detection_model3.ipynb
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/21664895/5060e7af-e280-46a6-9d6a-b89c0aaf8cd2/Untitled3.ipynb
