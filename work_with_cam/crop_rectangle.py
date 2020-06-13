import dlib
from PIL import Image
import numpy as np


def crop_rectangle(image):
    def find_face_rectangle(img):
        detector = dlib.get_frontal_face_detector()
        rectangles = detector(img, 1)
        faces = []
        for i, d in enumerate(rectangles):
            left = d.left()
            top = d.top()
            right = d.right()
            bottom = d.bottom()
            faces.append([left, top, right, bottom])

        if (faces[0][0] == 0 and faces[0][1] == 0 and faces[0][2] == 0 and faces[0][3] == 0) or i > 0:
            return None
        else:
            return faces

    def crop(img, rectangles):
        img_faces = []
        for rectangle in rectangles:
            img = Image.fromarray(img)
            img_face = img.crop((rectangle[0], rectangle[1], rectangle[2], rectangle[3]))
            img_face = img_face.resize((224, 224))
            img_faces.append(np.array(img_face))
        return img_faces

    small_faces = find_face_rectangle(image)
    images = crop(image, small_faces)
    return images
