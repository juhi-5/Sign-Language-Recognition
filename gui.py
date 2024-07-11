import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk, ImageOps
import numpy as np
import json
from tensorflow.keras.models import model_from_json

# Load model
def load_model():
    with open('model_json.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('SignLanguage.h5')
    return model

model = load_model()

# Define the labels (assuming the dataset includes A-Z and 0-9)
labels = [str(i) for i in range(1, 10)] + [chr(i) for i in range(65, 91)] 

# Function to process the image and predict
def predict(image_path):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB
    image = ImageOps.fit(image, (50, 50), Image.ANTIALIAS)  # Adjust the size to (50, 50)
    image = np.array(image)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_label = labels[np.argmax(prediction)]
    return predicted_label

# Function to upload image
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert('RGB')  # Convert to RGB
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        panel = Label(window, image=img)
        panel.image = img
        panel.grid(row=2, column=0, columnspan=2)
        label.config(text="Detected Sign: " + predict(file_path))

# Set up GUI
window = tk.Tk()
window.title("Sign Language Detection")

upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.grid(row=0, column=0, padx=10, pady=10)

label = Label(window, text="Detected Sign: None")
label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

window.mainloop()
