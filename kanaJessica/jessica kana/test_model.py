import cv2
import numpy as np
import tensorflow as tf
import sys

IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43

def load_and_predict(image_path, model):
    """Load image and make prediction using the provided model"""
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
            
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0  # Normalize
        img = img.reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)  # Add batch dimension
        
        # Predict
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        
        return predicted_class, confidence
    
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}") from e

def main():
    """Main entry point for command-line usage"""
    if len(sys.argv) != 2:
        sys.exit("Usage: python test_model.py image_path")

    image_path = sys.argv[1]
    
    # Load model with error handling
    model_path = "c:/New_folder/PYTHONAI/jesica/my_best_model.h5"
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        sys.exit(f"Failed to load model: {str(e)}")

    # Make prediction
    try:
        predicted_class, confidence = load_and_predict(image_path, model)
        print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%")
    except Exception as e:
        sys.exit(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
