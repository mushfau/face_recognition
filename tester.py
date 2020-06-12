import cv2
import os
import numpy as np
import tensorflow as tf
from face_system import face_detect, draw_boundary, get_normalized

model = tf.keras.models.load_model("models/face_model")
model.summary()

names_ = {}
folders = os.listdir("sample/")
for index, folder in enumerate(folders):
    names_[index] = folder

print(names_)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

webcam = cv2.VideoCapture(0)
cv2.namedWindow('Tester', cv2.WINDOW_AUTOSIZE)

while True:
    _, image = webcam.read()
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_boxes = face_detect(image2)
    if len(face_boxes):
        faces = get_normalized(image2, face_boxes)
        for i, face in enumerate(faces):
            face = np.array([face])
            predictions = probability_model.predict(face)
            name = names_[int(np.argmax(predictions[0]))]
            print(name)
            cv2.putText(image, name, (face_boxes[i][0], face_boxes[i][1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 3, (50, 50, 50), 3)
        draw_boundary(image, face_boxes)
    cv2.imshow('Tester', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
