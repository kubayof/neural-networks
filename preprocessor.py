# This file is used to extract face landmarks from images in data/selected folder
# Dumps nupmy array in binary and images with face landmarks
# array structure: [[imageTypeNumber, grayImageArray, imageLandmarksArray]]

import cv2
import dlib
import numpy as np
import PIL.Image
import os

IMAGES_BASE_DIR = 'data/selected'
IMAGES_DUMP_DIR = 'data/with-face-landmarks'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

imagesData = []
for imageTypeName in os.listdir(IMAGES_BASE_DIR):
    for imageFileName in os.listdir(os.path.join(IMAGES_BASE_DIR, imageTypeName)):
        imageFullPath = os.path.join(IMAGES_BASE_DIR, imageTypeName, imageFileName)
        image = PIL.Image.open(imageFullPath)
        np_image = np.array(image)
        rects = detector(np_image, 1)
        if len(rects) < 1:
            print("Cannot find face on image " + imageFullPath)
            continue
        # There is only one face in each image
        rect = rects[0]
        # Get the landmark points
        shape = predictor(np_image, rect)
        # Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        np_image_copy = np.copy(np_image)
        # Draw landmarks and save image
        for i, (x, y) in enumerate(shape):
            # Draw the circle to mark the keypoint
            cv2.circle(np_image_copy, (x, y), 1, (0, 0, 255), -1)

        imageWithLandmarks = PIL.Image.fromarray(np_image_copy)
        imageDir = os.path.join(IMAGES_DUMP_DIR, imageTypeName)
        os.makedirs(imageDir, exist_ok=True)
        imageWithLandmarks.save(os.path.join(imageDir, imageFileName))

        # Save images to common binary
        joined = np.array([int(imageTypeName), np_image, shape_np])
        imagesData.append(joined)

np_imagesData = np.array(imagesData)
np.save('data/images.npy', np_imagesData)
