from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import tensorflow

app = Flask(__name__)

# Load your trained Keras model
model = load_model('chest_xray_prediction.h5')  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  if 'file' not in request.files:
    return jsonify({'error': 'No file part'})

  file = request.files['file']
  if file.filename == '':
    return jsonify({'error': 'No selected file'})

  if file:
    try:
      # Save the uploaded file
      print(file)
      image_path = os.path.join('static', 'images', file.filename)
      file.save(image_path)

      # Preprocess the image
      image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale=1.0 / 255.0)
      image_generator = image_generator.flow_from_directory(
          'static/images', batch_size=1, target_size=(150, 150))

      # Make a prediction
      prediction = model.predict_generator(image_generator)

      # Get the prediction result
      result = prediction[0][0]
      if result > 0.5:
        result = 'Infected (Pneumonia)'
      else:
        result = 'Normal'

      return jsonify({'result': result})
    except Exception as e:
      return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
