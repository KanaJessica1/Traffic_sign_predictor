import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class TrafficSignPredictor:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.image_path = None
        self.setup_ui()

    def setup_ui(self):
        # Configure window
        self.root.title("Traffic Sign Predictor 🚦")
        self.root.geometry("500x600")
        self.root.configure(bg="#ecf0f1")  # Light background

        # Create frames
        self.header_frame = tk.Frame(self.root, bg="#2980b9")  # Blue header
        self.content_frame = tk.Frame(self.root, bg="#ecf0f1")
        self.footer_frame = tk.Frame(self.root, bg="#2980b9")

        # Header section
        self.header_label = tk.Label(
            self.header_frame,
            text="Traffic Sign Recognition System",
            font=("Helvetica", 18, "bold"),
            fg="white",
            bg="#2980b9"
        )
        self.header_label.pack(pady=15)

        # Content section
        self.model_label = tk.Label(
            self.content_frame,
            text="Model Status: Not Loaded",
            font=("Arial", 12),
            fg="black",
            bg="#ecf0f1"
        )
        self.model_label.pack(pady=10)

        self.image_label = tk.Label(
            self.content_frame,
            text="No Image Selected",
            font=("Arial", 12),
            fg="black",
            bg="#ecf0f1"
        )
        self.image_label.pack(pady=10)

        self.prediction_label = tk.Label(
            self.content_frame,
            text="Prediction: -",
            font=("Arial", 12),
            fg="black",
            bg="#ecf0f1"
        )
        self.prediction_label.pack(pady=10)

        # Buttons frame
        self.button_frame = tk.Frame(self.content_frame, bg="#ecf0f1")
        self.button_frame.pack(pady=20)

        self.load_model_button = tk.Button(
            self.button_frame,
            text="Load Model",
            command=self.load_model,
            bg="#e74c3c",  # Red
            fg="white",
            font=("Arial", 12, "bold"),
            width=15
        )
        self.load_model_button.pack(side=tk.LEFT, padx=10)

        self.load_image_button = tk.Button(
            self.button_frame,
            text="Load Image",
            command=self.load_image,
            bg="#27ae60",  # Green
            fg="white",
            font=("Arial", 12, "bold"),
            width=15
        )
        self.load_image_button.pack(side=tk.LEFT, padx=10)

        self.predict_button = tk.Button(
            self.button_frame,
            text="Predict",
            command=self.predict,
            bg="#9b59b6",  # Purple
            fg="white",
            font=("Arial", 12, "bold"),
            width=15
        )
        self.predict_button.pack(side=tk.LEFT, padx=10)

        # Footer section
        self.footer_label = tk.Label(
            self.footer_frame,
            text="Developed by NGATSING TAKAM FRANCK",
            font=("Arial", 10),
            fg="white",
            bg="#2980b9"
        )
        self.footer_label.pack(pady=10)

        # Layout the frames
        self.header_frame.pack(fill=tk.X)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.footer_frame.pack(fill=tk.X)

    def load_model(self):
        try:
            model_path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("HDF5 Files", "*.h5")]
            )
            if model_path:
                self.model = load_model(model_path)
                self.model_label.config(text=f"Model Status: Loaded ({model_path})", fg="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def load_image(self):
        try:
            self.image_path = filedialog.askopenfilename(
                title="Select Traffic Sign Image",
                filetypes=[("Image Files", ".png .jpg .jpeg")]
            )
            if self.image_path:
                self.image_label.config(text=f"Image Selected: {self.image_path}", fg="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def predict(self):
        if not self.model or not self.image_path:
            messagebox.showerror("Error", "Please load both model and image first!")
            return

        try:
            # Preprocess image
            img = cv2.imread(self.image_path)
            img = cv2.resize(img, (30, 30))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            predictions = self.model.predict(img)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions) * 100

            # Update UI with prediction
            self.prediction_label.config(
                text=f"Prediction: Class {predicted_class} - {confidence:.2f}% Confidence",
                fg="green" if confidence > 70 else "orange"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignPredictor(root)
    root.mainloop()
