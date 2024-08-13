import os
import pickle
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Layer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Define the custom Attention and TileLayer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return output

class TileLayer(Layer):
    def __init__(self, max_length, **kwargs):
        super(TileLayer, self).__init__(**kwargs)
        self.max_length = max_length

    def call(self, inputs):
        return tf.tile(inputs, [1, self.max_length, 1])

    def get_config(self):
        config = super(TileLayer, self).get_config()
        config.update({"max_length": self.max_length})
        return config

# Load the tokenizer and model
tokenizer_path = 'tokenizer.pkl'
model_path = 'model.keras'
features_path = 'features.pkl'

# Load the tokenizer
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the trained model with custom layers
caption_model = load_model(model_path, custom_objects={'Attention': Attention, 'TileLayer': TileLayer})

# Load features
with open(features_path, 'rb') as handle:
    features = pickle.load(handle)

# Set the maximum length of the caption
max_length = 34  # Adjust this if necessary

# Allowed extensions for the uploaded image
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_image(path, img_size=224):
    img = load_img(path, target_size=(img_size, img_size))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length, features):
    feature = features[image]
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        y_pred = model.predict([feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == 'endseq':
            break

    return in_text.replace('startseq', '').replace('endseq', '').strip()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Read the image and predict caption
            image = filename
            caption = predict_caption(caption_model, image, tokenizer, max_length, features)

            return render_template('result.html', caption=caption, image_url=url_for('uploaded_file', filename=filename))

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
