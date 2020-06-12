import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_detect(input_image):
    faces_ = faceCascade.detectMultiScale(input_image, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    return faces_


def draw_boundary(input_image, faces_):
    for (x, y, w, h) in faces_:
        cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)


def cut_faces(input_image, face_boxes_):
    faces_ = []
    for (x, y, w, h) in face_boxes_:
        margin = int(0.2 * w / 2)
        faces_.append(input_image[y: y + h, x + margin:x + w - margin])
    return faces_


def get_intensity_normalized(input_faces):
    faces_ = []
    for face in input_faces:
        is_rbg = len(face.shape) == 3
        if is_rbg:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        faces_.append(cv2.equalizeHist(face))
    return faces_


def get_resized(input_faces):
    dimension = (50, 50)
    faces_ = []
    for face in input_faces:
        resized_face = cv2.resize(face, dimension, interpolation=cv2.INTER_AREA)
        faces_.append(resized_face)
    return faces_


def get_normalized(input_image, face_boxes_):
    faces_ = cut_faces(input_image, face_boxes_)
    faces_ = get_intensity_normalized(faces_)
    faces_ = get_resized(faces_)
    return faces_
