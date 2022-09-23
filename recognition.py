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
    detect = detector.detectMultiScake(img_grey, )