from deepface import DeepFace
import cv2
import math
import argparse
import requests
import json

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                 104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes


parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

faceNet = cv2.dnn.readNet(faceModel, faceProto)


video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20
while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]-padding):
                     min(faceBox[3]+padding, frame.shape[0]-1), max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

        cv2.imwrite('data/test.jpg', face)

        # displaying the adjusted image with added facebox and text, font, BRG (Blue,green,red ~ 0-255), thickness
        cv2.putText(resultImg,"Ready to analyze!", (
            faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_4)
        cv2.imshow("BiasedAI POC", resultImg)

#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) 
#identical to the line above passing nothing as 2nd argument will find everything
try: 
    demography = DeepFace.analyze("data/test.jpg",['age','gender','race']) 
    print(f'{demography["age"]}, {demography["gender"]},{demography["dominant_race"]}')
except Exception as e:
    print(e)

 # Post data
try:
    api = "http://localhost:8083/PersonData"
    params = {"age": demography["age"],
    "gender": demography["gender"],
    "race":demography["dominant_race"],
    }
    r = requests.post(url= api,data=params)
    print(r.text)

except requests.exceptions.RequestException as e:  # This is the correct syntax
    print(e)

