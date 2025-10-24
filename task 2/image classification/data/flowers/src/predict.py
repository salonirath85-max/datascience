import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("outputs/model.h5")

def predict_image(image_path):
    img = Image.open(image_path).resize((180, 180))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    print(f"Predicted class index: {class_idx}")

# Example usage
predict_image("data/test_image.jpg")