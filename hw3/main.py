import cv2
import mediapipe as mp
import numpy as np
from fer import FER

# Распознавание лиц
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_model.yml')
your_label = 0  # поменяй на свой label

# Распознавание эмоций
emotion_detector = FER(mtcnn=True)

# Настройка MediaPipe для рук
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
tip_ids = [4, 8, 12, 16, 20]  # кончики пальцев

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Обнаружение лиц
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Обнаружение жестов
    results = hands.process(img_rgb)
    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            fingers = []

            # Большой палец
            if hand_lms.landmark[4].x < hand_lms.landmark[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # Остальные пальцы
            for id in range(1, 5):
                tip_y = hand_lms.landmark[tip_ids[id]].y
                pip_y = hand_lms.landmark[tip_ids[id] - 2].y
                fingers.append(1 if tip_y < pip_y else 0)

            finger_count = sum(fingers)

    # Обработка каждого лица
    for (x, y, w, h) in faces:
        face_resized = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        label, confidence = recognizer.predict(face_resized)

        # Цвет и имя
        color = (0, 255, 0) if label == your_label and confidence < 80 else (0, 0, 255)
        name = "BOSS" if label == your_label and confidence < 80 else "unknown"

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, name, (x + 6, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ---- ПОКАЗЫВАТЬ ЭМОЦИЮ ТОЛЬКО ЕСЛИ ПОДНЯТО 3 ПАЛЬЦА ----
        if finger_count == 3:
            face_img = img[y:y + h, x:x + w]
            try:
                emotion_result = emotion_detector.top_emotion(face_img)
                emotion_label = emotion_result[0] if emotion_result[0] else "unknown"
            except Exception as e:
                print("Ошибка распознавания эмоции:", e)
                emotion_label = "unknown"

            cv2.putText(img, f"Emotion: {emotion_label}", (x + 6, y - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Отображение действий по количеству пальцев
    if len(faces) > 0 and finger_count > 0:
        x_face, y_face = faces[0][0], faces[0][1]

        if finger_count == 1:
            cv2.putText(img, "Ruslan", (x_face, y_face - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif finger_count == 2:
            cv2.putText(img, "Khairullov", (x_face, y_face - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif finger_count == 3:
            cv2.putText(img, "Action: Zoom", (x_face, y_face - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Показ кадра
    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()