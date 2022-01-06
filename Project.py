import numpy as np
import cv2
from datetime import datetime
from datetime import date
import gpiozero

face_cascade = cv2.CascadeClassifier('D:\\CV2\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('D:\\CV2\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml')
capture = cv2.VideoCapture(0)


# shot_idx = 0


def detect(gray, face_cascade):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return []
    faces[:, 2:] += faces[:, :2]
    return faces


def draw_rects(gray, faces, color):
    for x1, y1, x2, y2 in faces:
        cv2.rectangle(gray, (x1, y1), (x2, y2), color, 2)


def main():
    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = detect(gray, face_cascade)
        vis = frame.copy()
        draw_rects(vis, faces, (0, 255, 0))
        now = datetime.now()
        current_time = now.strftime("%H%M%S")
        today = date.today()
        d = today.strftime("%d%m%Y")

        if len(faces) == 1:
            fn = 'D:\\python projects\\Final Year Project\\image data\\shot%s_%s.jpg' % (d, current_time)
            cv2.imwrite(fn, frame)
            
            # shot_idx += 1
        elif len(faces) > 1:
            print("Error-multiple faces detected")

        cv2.imshow('frame', vis)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break


if __name__ == '__main__':
    print(__doc__)
    main()
    capture.release()
    cv2.destroyAllWindows()
