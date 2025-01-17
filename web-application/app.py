from flask import Flask, request, render_template, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# prevent DecompressionBombError during Dev.
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


app = Flask(__name__)

# Load the trained model
MODEL_PATH = "MyModel.keras"
model = tf.keras.models.load_model(MODEL_PATH)


# class labels
CLASS_NAMES = [
                'apple','banana','beetroot','bell pepper',
                'cabbage','capsicum','carrot','cauliflower',
                'chilli pepper','corn','cucumber','eggplant','garlic','ginger','grapes','jalepeno',
                'kiwi','lemon','lettuce','mango','onion','orange','paprika','pear','peas','pineapple','pomegranate','potato',
                'raddish','soy beans','spinach','sweetcorn','steetpotato','tomato','turnip','watermelon']

# preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)  # Load and resize the image
    img_array = img_to_array(img) / 255.0  # Convert to array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        # Save the uploaded file in static/uploads
        upload_folder = os.path.join('static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # preprocess the image and predict
        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]

        # Generate the image URL and render the result
        image_url = url_for('static', filename=f'uploads/{file.filename}')
        return render_template('result.html', label=predicted_class, image_url=image_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)