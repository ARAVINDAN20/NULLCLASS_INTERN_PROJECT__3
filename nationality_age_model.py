import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Paths to data
celeba_dir = r"E:\Emotion Detection\archive\img_align_celeba\img_align_celeba"
attributes_file = r"E:\Emotion Detection\archive\list_attr_celeba.csv"

# Load attributes
attributes = pd.read_csv(attributes_file)

# Add synthetic age and nationality columns
np.random.seed(42)
attributes['age'] = np.random.randint(10, 61, size=len(attributes))
nationalities = ['Indian', 'United States', 'African', 'Other']
attributes['nationality'] = np.random.choice(nationalities, size=len(attributes))

# Create nationality mapping
nationality_mapping = {nat: i for i, nat in enumerate(nationalities)}
attributes['nationality'] = attributes['nationality'].map(nationality_mapping)

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    return img_array

# Prepare data
num_samples = 10000  # Adjust this number based on your available memory
sample_df = attributes.sample(n=num_samples, random_state=42)

X = np.array([load_and_preprocess_image(os.path.join(celeba_dir, img_id)) for img_id in sample_df['image_id']])
y_nationality = np.array(sample_df['nationality'])
y_age = np.array(sample_df['age'])

# Split the data
X_train, X_test, y_nationality_train, y_nationality_test, y_age_train, y_age_test = train_test_split(
    X, y_nationality, y_age, test_size=0.2, random_state=42)

# Create the model using Functional API
inputs = Input(shape=(128, 128, 3))
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
nationality_output = Dense(len(nationalities), activation='softmax', name='nationality_output')(x)
age_output = Dense(1, activation='linear', name='age_output')(x)

model = Model(inputs=inputs, outputs=[nationality_output, age_output])

# Compile the model
model.compile(optimizer='adam',
              loss={'nationality_output': 'sparse_categorical_crossentropy', 'age_output': 'mse'},
              loss_weights={'nationality_output': 1.0, 'age_output': 0.5},
              metrics={'nationality_output': 'accuracy', 'age_output': 'mae'})

# Train the model
history = model.fit(
    X_train,
    {'nationality_output': y_nationality_train, 'age_output': y_age_train},
    validation_data=(X_test, {'nationality_output': y_nationality_test, 'age_output': y_age_test}),
    epochs=10,
    batch_size=32
)

# Save the model
model.save("nationality_age_model.h5")

print("Model training completed and saved as 'nationality_age_model.h5'")


# completed🎀☺️