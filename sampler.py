import cv2
import os

from face_system import face_detect, draw_boundary, get_normalized

name = ""

while name == "":
    name = input("Enter your name or 'Q' to exit: ")
    if name in ['Q', 'q']:
        exit(0)

print('you entered: ', name)
directory = "sample/" + name.lower()

webcam = cv2.VideoCapture(0)
cv2.namedWindow('Sampler', cv2.WINDOW_AUTOSIZE)
cv2.waitKey(1000)

if not os.path.exists(directory):
    os.mkdir(directory)
    count = 0
    timer = 0
    while count < 10:
        _, image = webcam.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_boxes = face_detect(image)
        if len(face_boxes) and timer % 900 == 50:
            faces = get_normalized(image, face_boxes)
            print("saving image ", count)
            cv2.imwrite(directory + '/' + str(count) + '.jpg', faces[0])
            count += 1
        draw_boundary(image, face_boxes)
        cv2.imshow('Sampler', image)
        cv2.waitKey(50)
        timer += 50
else:
    print("This person exists")

webcam.release()
cv2.destroyAllWindows()
