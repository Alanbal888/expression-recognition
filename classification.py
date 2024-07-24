import tensorflow as tf
import numpy as np
import cv2


class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
def format_image(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
    image = image.astype(float) / 255
    image = np.expand_dims(image, axis=0)
    return image


class ExpressionClassifier:
    model = None

    def __init__(self):
        self.model = tf.keras.models.load_model('models/expression_recognition.h5')

    def classify(self, img):
        formatted_image = format_image(img)
        validation_pred_probs = self.model.predict(formatted_image)
        validation_pred_labels = np.argmax(validation_pred_probs, axis=1)
        return class_names[validation_pred_labels[0]]
