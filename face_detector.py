import cv2
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from classification import ExpressionClassifier

detections = []

import cv2
import os
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 255, 0)
thickness = 2

def draw_faces(img, detections):
    for detection in detections:
        for (x, y, w, h) in detection:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_emotion(img, face_detection, emotion="None Detected"):
    for (x, y, w, h) in face_detection:
        cv2.putText(img, emotion, (x, y + h + 30), font, fontScale, color, thickness, cv2.LINE_AA)




class FaceDetector:
    face_classif = None
    lateral_face_classif = None
    eyes_classif = None

    expression_classifier = None
    emotion = None

    def __init__(self):
        self.face_classif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.lateral_face_classif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eyes_classif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        self.expression_classifier = ExpressionClassifier()

    def body_part_recognition(self, img, body_part):
        global faces1
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clasif = None
        match body_part:
            case "face":
                clasif = self.face_classif
            case "lateral":
                clasif = self.lateral_face_classif
            case "eyes":
                clasif = self.eyes_classif

        detection = clasif.detectMultiScale(gray_img,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(30, 30),
                                            maxSize=(200, 200))
        detections.append(detection)

    def draw_face_detections(self, frame):

        global detections

        faces1_process = Thread(target=self.body_part_recognition, args=[frame, "face"])
        faces1_process.start()
        faces1_process.join()

        new_emotion = "None"
        if len(detections) > 0:
            face_detection = detections[0]
            face_only = None
            for (x, y, w, h) in face_detection:
                face_only = frame[y:y + h, x:x + w]
            if face_only is not None:
                new_emotion = self.expression_classifier.classify(face_only)

        draw_faces(frame, detections)
        if new_emotion != self.emotion:
            self.emotion = new_emotion

        draw_emotion(frame, detections[0], self.emotion)
        cv2.imshow('frame', frame)
        detections = []
