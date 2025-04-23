import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import accuracy_score

model = keras.models.load_model('mnist_model.h5')

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Skaliraj slike u raspon [0, 1]
x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

# Pretvori labele u one-hot encoding
y_test_s = keras.utils.to_categorical(y_test, 10)

# predict
y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)

# loše klasificirane slike
misclassified_indices = np.where(y_pred_classes != y_test)[0]

plt.figure(figsize=(10, 4))
for i, index in enumerate(misclassified_indices[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[index], cmap='gray')
    true_label = y_test[index]
    predicted_label = y_pred_classes[index]
    plt.title(f"True: {true_label}, Pred: {predicted_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()