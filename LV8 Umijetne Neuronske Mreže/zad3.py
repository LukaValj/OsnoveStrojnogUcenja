import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image, ImageOps

model = keras.models.load_model('mnist_model.h5')

image_path = 'broj.png'

def process_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img = ImageOps.invert(img)
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_number(image_path):
    processed_image = process_image(image_path)

    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return predicted_class

predicted_number = predict_number(image_path)

img = Image.open(image_path)
plt.imshow(img, cmap='gray')
plt.title(f"Predviđeni broj: {predicted_number}")
plt.axis('off')
plt.show()