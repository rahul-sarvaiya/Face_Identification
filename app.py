from collections import Counter
import pathlib
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import face_recognition as fr
import os
import cv2
import face_recognition
from time import sleep
from ultralytics import YOLO 
from ultralyticsplus import render_result

model = YOLO("exports/helmet_detection.pt")

def Objectdetection(im):
    for dirpath, dnames, fnames in os.walk("./"+im):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                image = im+"/" + f
                results =model.predict(source=image,show=True, conf = 0.5)
                render = render_result(model=model, image=image, result=results[0])
                render.show()


def Objectdetection2(im):
    flag=0
    image = im
    results =model.predict(source=image,show=True, conf = 0.5)
    render = render_result(model=model, image=image, result=results[0])
    #render.show()
    if bool(results[0].boxes):
        flag=1
    else:
        flag=0
    return flag


def get_encoded_faces():
    encoded = {}
    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
                print("encoded faces",encoded)

    return encoded


def unknown_image_encoded(img):
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding

def classify_face(im):
    faces = get_encoded_faces()    #call get faces
    faces_encoded = list(faces.values()) #assigning value
    known_face_names = list(faces.keys()) #assigning keys

    img = cv2.imread(im, 1) #person img
    mg = cv2.resize(img, (0, 0), fx=0.5, fy=0.5) 
    #img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img) #detecting location of face into image
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations) #identification

    
    face_names = []
    face_helmet = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding) #comaring image 
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            print("known_face_names",known_face_names)

        face_names.append(name) #appending names

        flag=Objectdetection2(im) #call helmet detection function

        if flag==1:
            face_helmet.append("Helmet Detected")
        else:
            face_helmet.append("Helmet Not Detected")


        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    # Display the resulting image
    while True:
        # return face_names 
        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return (face_names,face_helmet)


def Markattendance(im):
    
    f_name=[]
    for dirpath, dnames, fnames in os.walk("./"+im):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                f_name.append(classify_face(im+"/" + f)) #calling classify_face function
    return f_name

  
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/attendancemarked',methods=['POST'])
def attendancemarked():
    int_features = [str(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    output = int_features[0]
    output2=pathlib.Path(format(output)).name
    return render_template('index.html', prediction_text=Markattendance(format(output2)))

# @app.route('/detection')
# def detection():
#     return render_template('detection.html')

# @app.route('/detectiondone',methods=['POST'])
# def detectiondone():
#     int_features = [str(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     output = int_features[0]
#     output2=pathlib.Path(format(output)).name
#     return render_template('detection.html', prediction_text=Objectdetection(format(output2)))


if __name__ == "__main__":
    app.run(debug=True)