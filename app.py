from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("BestModel.keras")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file temporarily to process it
    filename = secure_filename(file.filename)
    file_path = f'./uploads/{filename}'
    file.save(file_path)

    # Load and preprocess the image
    img = Image.open(file_path)
    img = img.resize((256, 256))  # Resize to match model input size
    img = np.array(img)  # Convert image to a numpy array
    if img.shape[-1] == 3:  # Check if image is RGB
        # Convert to grayscale if needed
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize image if model expects [0,1]

    # Make a prediction
    prediction = model.predict(img)
    # Assuming classification task
    prediction_value = prediction[0][0]  # Get the scalar value from the result

    # Apply threshold for classification
    if prediction_value < 0.5:
        result = "Brain Tumor Detected"
    else:
        result = "No Brain Tumor"

    # Return the prediction as a response
    return jsonify({'prediction': str(result), 'confidence': float(prediction_value)})


if __name__ == '__main__':
    app.run(debug=True)
