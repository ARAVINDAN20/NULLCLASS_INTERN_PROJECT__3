import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
from sklearn.cluster import KMeans

# Load models
nationality_age_model = load_model('nationality_age_model.h5', compile=False)
emotion_model = load_model('model.h5')
age_gender_model = load_model('best_model.keras')

nationalities = ['Indian', 'United States', 'African', 'Other']
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Simple color dictionary
COLOR_DICT = {
    'red': ([0, 0, 100], [80, 80, 255]),
    'green': ([0, 100, 0], [80, 255, 80]),
    'blue': ([100, 0, 0], [255, 80, 80]),
    'yellow': ([0, 100, 100], [80, 255, 255]),
    'orange': ([0, 50, 100], [80, 150, 255]),
    'purple': ([50, 0, 50], [150, 80, 150]),
    'pink': ([100, 0, 100], [255, 80, 255]),
    'brown': ([0, 0, 0], [80, 80, 80]),
    'white': ([200, 200, 200], [255, 255, 255]),
    'black': ([0, 0, 0], [50, 50, 50])
}


class NationalityAgeEmotionGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Nationality, Age, and Emotion Predictor")
        self.master.geometry("600x700")
        self.master.configure(bg='#f0f0f0')
        self.create_widgets()
        self.image_path = None

    def create_widgets(self):
        self.button_frame = tk.Frame(self.master, bg='#f0f0f0')
        self.button_frame.pack(pady=20)

        self.upload_btn = tk.Button(self.button_frame, text="Upload Image", command=self.upload_image, bg='#4CAF50',
                                    fg='white', font=("Arial", 12))
        self.upload_btn.grid(row=0, column=0, padx=10)

        self.predict_btn = tk.Button(self.button_frame, text="Predict", command=self.predict_image, bg='#008CBA',
                                     fg='white', font=("Arial", 12), state=tk.DISABLED)
        self.predict_btn.grid(row=0, column=1, padx=10)

        self.cancel_btn = tk.Button(self.button_frame, text="Cancel", command=self.cancel, bg='#f44336', fg='white',
                                    font=("Arial", 12))
        self.cancel_btn.grid(row=0, column=2, padx=10)

        self.image_label = tk.Label(self.master, bg='#f0f0f0')
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(self.master, text="", font=("Arial", 12), justify=tk.LEFT, bg='#f0f0f0',
                                     wraplength=500)
        self.result_label.pack(pady=10)

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            self.display_image(self.image_path)
            self.predict_btn['state'] = tk.NORMAL

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img

    def predict_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Predict nationality and age
        img_nationality_age = load_and_preprocess_image(self.image_path, (128, 128))
        img_nationality_age = np.expand_dims(img_nationality_age, axis=0)
        nationality_pred, age_pred = nationality_age_model.predict(img_nationality_age)

        predicted_nationality = nationalities[np.argmax(nationality_pred[0])]
        predicted_age = int(age_pred[0][0])

        # Predict emotion
        img_emotion = load_and_preprocess_image(self.image_path, (48, 48), grayscale=True)
        img_emotion = np.expand_dims(img_emotion, axis=0)
        emotion_pred = emotion_model.predict(img_emotion)
        predicted_emotion = emotions[np.argmax(emotion_pred[0])]

        # Calculate confidence
        confidence = np.max(nationality_pred[0]) * 100

        result_text = f"Predicted Nationality: {predicted_nationality}\n"
        result_text += f"Confidence: {confidence:.2f}%\n"

        # Additional information based on nationality and age constraints
        if 10 <= predicted_age <= 60:
            if predicted_nationality == "Indian":
                result_text += f"Age: {predicted_age}\n"
                result_text += f"Emotion: {predicted_emotion}\n"
                result_text += f"Dress Color: {self.get_dominant_color(self.image_path)}\n"
            elif predicted_nationality == "United States":
                result_text += f"Age: {predicted_age}\n"
                result_text += f"Emotion: {predicted_emotion}\n"
            elif predicted_nationality == "African":
                result_text += f"Emotion: {predicted_emotion}\n"
                result_text += f"Dress Color: {self.get_dominant_color(self.image_path)}\n"
            else:
                result_text += f"Emotion: {predicted_emotion}\n"
        else:
            result_text += "Age prediction is outside the valid range (10-60 years).\n"
            result_text += f"Emotion: {predicted_emotion}\n"

        self.result_label.config(text=result_text)

    def cancel(self):
        self.image_path = None
        self.image_label.config(image='')
        self.result_label.config(text="")
        self.predict_btn['state'] = tk.DISABLED

    def get_dominant_color(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        clt = KMeans(n_clusters=3)
        clt.fit(image)

        dominant_color = clt.cluster_centers_[0].astype(int)
        return self.closest_color(dominant_color)

    def closest_color(self, rgb):
        r, g, b = rgb
        color_diffs = []
        for color_name, (low, high) in COLOR_DICT.items():
            if all(low[i] <= val <= high[i] for i, val in enumerate([r, g, b])):
                return color_name
        return "Unknown"


def load_and_preprocess_image(image_path, target_size, grayscale=False):
    if grayscale:
        img = load_img(image_path, target_size=target_size, color_mode='grayscale')
        img_array = img_to_array(img).reshape(target_size + (1,))
    else:
        img = load_img(image_path, target_size=target_size, color_mode='rgb')
        img_array = img_to_array(img)
    return img_array / 255.0


if __name__ == "__main__":
    root = tk.Tk()
    app = NationalityAgeEmotionGUI(root)
    root.mainloop()
