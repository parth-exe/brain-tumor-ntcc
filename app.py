import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model = load_model('Deployment/saved-models')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
    dictionary = {
        0:'Meningioma',
        1:'Glioma',
        2:'Pituitary Tumour'
    }
    classNo = np.argmax(classNo)
    return dictionary[classNo]


def getResult(img):
    image=cv2.imread(img)
    grayscale = image[:, :, 0]
    img = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
    resized = tf.image.resize(img, (512,512))
    predic = model.predict(np.expand_dims(resized/255,0))

    return predic


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)