# Traffic Sign Predictor Application

## Overview
This application uses a neural network model to classify traffic signs. It is trained on a traffic sign dataset and can predict the type of traffic sign from an image. A user-friendly Tkinter GUI allows users to upload an image and view the prediction.

## Features
- **Model Loading**: Users can load a trained model (`.h5` file).
- **Image Loading**: Users can load an image of a traffic sign for prediction.
- **Prediction**: The model predicts the class of the sign and displays the confidence level.
- **Error Handling**: The app provides error messages if model or image loading fails.

## Design Enhancements
- **Color Scheme**: The background has been updated to a light color (`#ecf0f1`), with the header and footer in a bold blue (`#2980b9`) for contrast. The buttons have vibrant colors to make them stand out.
- **Layout**: The window size is set to `500x600` to better accommodate the content. Button widths are adjusted for uniformity, and padding is added to improve spacing between elements.
- **Typography**: The font has been changed to `Helvetica`, and font sizes have been optimized for better readability.
- **Spacing**: Additional padding and margins are added to buttons and labels to create a cleaner and more organized interface.

## File Structure
- **traffic_sign_predictor.py**: Python script for the application.
- **best_model.h5**: The pre-trained model saved in HDF5 format.
- **images/**: Directory containing sample traffic sign images for testing.
  
## Installation & Setup

### 1. Install Dependencies
- **TensorFlow**:
    ```bash
    pip install tensorflow
    ```
- **Tkinter** (if not already installed):
    ```bash
    sudo apt-get install python3-tk
    ```

### 2. Clone the Repository
- Clone the project repository to your local machine:
    ```bash
    git clone KanaJessica1://github.com
    ```

### 3. Set Up Virtual Environment (Optional)
- Create a virtual environment for the project:
    ```bash
    python -m venv venv
    ```
- Activate the virtual environment:
    - On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

### 4. Install Required Dependencies
- Once the virtual environment is activated, install the necessary dependencies:
    ```bash
    pip install tensorflow
    pip install tk
    ```

## Running the Application

1. **Load Model**: Click on "Load Model" to select and load the pre-trained `.h5` model.
2. **Load Image**: Click on "Load Image" to select the traffic sign image you wish to classify.
3. **Predict**: Click on "Predict" to classify the selected image. The app will display the predicted class and confidence level.

## Developer Information
- **Developed by**: jessica kana

Let me know if you need any further changes!
