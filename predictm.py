import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from keras.models import load_model
import os


def get_cropped_image_if_2_eyes(image_path):
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 1:
            return roi_color


model = load_model('classify_model.h5')

class_names = ['Karim Benzema', 'De Bruyen', 'Harry Kane', 'Robert Lewandowski', 'Kilian Mbappe', 'Leo Messi',
               'Neymar Jr.', 'Cristiano Ronaldo', 'Mohamed Salah', 'Luis Suarez',
               'Sunil Chetri']


def classify_image(file_path):
    cv2.imwrite("test123.jpg", get_cropped_image_if_2_eyes(file_path))

    path = "test123.jpg"
    # cropped_image = get_cropped_image_if_2_eyes(path)
    # matplotlib.image.imsave('name.jpg', cropped_image)

    # plt.imshow(cv2.imread("name.jpg"))
    # path="name.jpg"
    img = tf.keras.utils.load_img(
        path, target_size=(256, 256)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    result = {
        'class': class_names[np.argmax(score)],
        'class_probability': 100 * np.max(score)
    }
    try:
        os.remove('test123.jpg')
    except:
        pass

    return result
