from flask import Flask, request, render_template
import os
from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

app = Flask(__name__)

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Function to load and preprocess image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Function to predict and return the prediction
def predict_image(img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0][0]
    confidence = round(decoded_predictions[2] * 100)  # Round to the nearest whole number
    return (decoded_predictions[0], decoded_predictions[1], confidence)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Ensure the uploads directory exists
            uploads_dir = os.path.join('static', 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)  # Create the directory if it doesn't exist
            
            # Save the file to the uploads directory
            img_path = os.path.join(uploads_dir, file.filename)
            file.save(img_path)
            prediction = predict_image(img_path)
            return render_template('result.html', prediction=prediction, image_filename=file.filename)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)