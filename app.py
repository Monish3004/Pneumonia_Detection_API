import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import pydicom
import io
import cv2 as cv
import matplotlib.pyplot as plt
from flask_cors import CORS  # Importing CORS

app = Flask(__name__)

CORS(app)  # This will enable CORS for all domains

# Define a dictionary of custom objects (if any)
custom_objects = {
    "mse": tf.keras.losses.MeanSquaredError()
}

# Load the model with custom objects
model = tf.keras.models.load_model("pneumonia_model.h5", custom_objects=custom_objects)

# Image preprocessing function
def preprocess_image(image, input_size=(244, 244)):
    image_resized = cv.resize(image, input_size)
    image_resized = image_resized.astype("float32") / 255.0  # Normalize the image
    image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension
    return image_resized

# Function to process DICOM images
def process_dicom(dicom_file):
    # Read the DICOM file
    dicom_data = pydicom.dcmread(dicom_file)
    
    # Extract the pixel data from the DICOM object
    image_data = dicom_data.pixel_array
    
    # Convert to PIL Image for further processing
    image = Image.fromarray(image_data)
    
    return image



# Root route
@app.route("/", methods=["GET"])
def home():
    return "Hi,Welcome to the Pneumonia Detection API!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        print("No file uploaded")  # Log error if no file is found
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    print(f"Received file: {file.filename}")  # Log file name

    # Check if the file is a DICOM file or image
    try:
        if file.filename.endswith(".dcm"):
            # Process DICOM image
            image = process_dicom(file)
            
            # Preprocess the image to match the format you specified
            processed_image = preprocess_image(np.array(image))

            # Perform inference on the image (image_array)
            output = model.predict(processed_image)  # Use just the processed image for prediction
            #print(output)

            # Extract prediction information
            predicted_box = output[1][0]  # Box coordinates
            predicted_label = output[0][0][1]  # Predicted label (probability for positive class)
            #print(predicted_box)
            # Only visualize if the prediction is positive
            if predicted_label > 0.5:
                a=[]
                for i in predicted_box:
                    a.append(float(i))
                return jsonify({"message": "Prediction is positive","coor":a}), 200
            else:
                return jsonify({"message": "Prediction is negative"}), 200

        else:
            # Process regular image (JPEG, PNG, etc.)
            image = Image.open(file.stream).convert("RGB")
            processed_image = preprocess_image(np.array(image))

            # Perform inference
            output = model.predict(processed_image)  # Use just the processed image for prediction
            #print(output)

            # Extract prediction information
            predicted_box = output[1][0]  # Box coordinates
            predicted_label = output[0][0][1]  # Predicted label (probability for positive class)
            #print(predicted_box)
            
            # Only visualize if the prediction is positive
            if predicted_label > 0.5:
                a=[]
                for i in predicted_box:
                    a.append(float(i))
                return jsonify({"message": "Prediction is positive","coor":a}), 200
            else:
                return jsonify({"message": "Prediction is negative"}), 200

    except Exception as e:
        print(f"Error processing file: {str(e)}")  # Log error during file processing
        return jsonify({"error": "Invalid file type", "details": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
