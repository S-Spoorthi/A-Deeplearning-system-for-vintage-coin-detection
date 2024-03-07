import os
from flask import Flask, render_template, request, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = load_model('C:\\Users\\spoor\\Documents\\flask_proj\\mobilenet_model.h5')  
target_size = (96, 96)  


def predict(image_path):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image data

    prediction = model.predict(img_array)
    classes = [
    'Andorra_2019',
    'Common',
    'Lithuania_2021',
    'Monaco_2015',
    'Monaco_2016',
    'Monaco_2017',
    'Monaco_2018',
    'Monaco_2019',
    'SanMarino_2004',
    'SanMarino_2005',
    'Vatican_2004',
    'Vatican_2005',
    'Vatican_2006'
]  
    predicted_class = classes[np.argmax(prediction)]
    probability = np.max(prediction)
    return predicted_class, probability, img


@app.route('/')
def index():
    return render_template('index.html')

"""
@app.route('/predict', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction, img = predict(file_path)
            img_url = '/' + os.path.join(app.config['UPLOAD_FOLDER'], filename)  
            print("img_url:", img_url)
            return render_template('index.html', prediction=prediction, img_url=img_url)

    return render_template('index.html', prediction=None, img_url=None)
"""

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction, probability, img = predict(file_path)
            img_url = url_for('uploaded_file', filename=filename)
            print("img_url:", img_url)
            return render_template('index.html', prediction=prediction, probability=probability, img_url=img_url)

    return render_template('index.html', prediction=None, probability=None, img_url=None)


if __name__ == '__main__':
    app.run(debug=True)
