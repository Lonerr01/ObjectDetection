import cv2

car_cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_src)

human_cascade_src = 'haarcascade_fullbody.xml'
human_cascade = cv2.CascadeClassifier(human_cascade_src)

video_src = 'video.avi'
cap = cv2.VideoCapture(video_src)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Araba tespiti
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # İnsan tespiti
    humans = human_cascade.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in humans:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Detection', img)

    if cv2.waitKey(33) == 27:
        break

cap.release()
cv2.destroyAllWindows()
