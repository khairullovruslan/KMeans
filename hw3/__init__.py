import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
dataset_path = "faces_dataset"
images = []
labels = []

for file in os.listdir(dataset_path):
    if file.endswith(".jpg"):
        path = os.path.join(dataset_path, file)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        images.append(image)
        labels.append(0)

recognizer.train(images, np.array(labels))
recognizer.save('face_model.yml')
print("Модель обучена на", len(images), "изображениях")