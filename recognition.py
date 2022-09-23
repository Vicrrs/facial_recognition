import cv2

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("lbph_classifier.yml")
height, width = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
cam = cv2.VideoCapture(0)

while True:
    ok, frame = cam.read()
    img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect = detector.detectMultiScale(img_grey, scaleFactor=1.5, minSize=(30,30))
    for (x, y, w, h) in detect:
        img_face = cv2.resize(img_grey[y:y+w, x:x+h], (height, width))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        id, confidence = recognizer.predict(img_face)
        name = ""
        if id == 1:
            name = 'Heitor'
        elif id == 2:
            name = 'Victor'
        cv2.putText(frame, name, (x, y+(w+30)), font, 2,(0, 0, 255))
        cv2.putText(frame, str(confidence), (x, y+(h+50)), font, 1,(0, 0, 255))

    cv2.imshow("Face", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
