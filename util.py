import joblib
import json
import numpy as np
import base64
import cv2
import tensorflow as tf

from tensorflow import keras
from wavelet import w2d

__model = None
class_names = ['benzema', 'de_bruyen', 'kane', 'lewandowski', 'mbappe', 'messi', 'neymar', 'ronaldo', 'salah', 'suarez',
               'sunil']
img_height = 256
img_width = 256


def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []
    for img in imgs:
        path = "download (2).jpg"
        # cropped_image = get_cropped_image_if_2_eyes(path)
        # matplotlib.image.imsave('name.jpg', cropped_image)

        # plt.imshow(cv2.imread("name.jpg"))
        # path="name.jpg"
        img = tf.keras.utils.load_img(
            path, target_size=(img_height, img_width)
        )

        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        result.append({
            'class': class_names[np.argmax(score)],
            'class_probability': 100 * np.max(score)
        })

    return result


def load_saved_model():
    global __model
    if __model is None:
        with open('classify_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    # print("loading saved artifacts...done")


def get_cv2_image_from_b64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.units)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_b64_string(image_base64_data)

    gray = cv2.cvtcolor(img, cv2.color_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces


def get_b64_test_image():
    with open("b64img.txt") as f:
        return f.read()


if __name__ == "__main__":
    load_saved_model()
