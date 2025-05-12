import cv2
import os

dataset_path = "faces_dataset"
user_name = "Ruslan"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cap = cv2.VideoCapture(0)
count = 0

print("Нажимайте Пробел, чтобы сделать фото. Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 200))

    cv2.imshow('Сбор данных', frame)

    key = cv2.waitKey(1)
    if key == 32:
        filename = f"{dataset_path}/{user_name}_{count}.jpg"
        cv2.imwrite(filename, resized)
        print(f"Сохранено: {filename}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()