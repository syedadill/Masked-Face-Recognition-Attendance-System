import os
import pickle
import cv2
import cvzone
import face_recognition
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import numpy as np
from datetime import datetime
import dlib
from imutils import face_utils

# Load the Haar cascade classifier for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the facial landmarks predictor for detecting facial landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL':"https://realtimefaceattendance-26fa3-default-rtdb.firebaseio.com/",
    'storageBucket':"realtimefaceattendance-26fa3.appspot.com"
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('resources/background.png')

# Importing modes
folderModePath = 'resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []

for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# Load the encoding file
print("Loading the encoded files...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds= pickle.load(file)
file.close()
encodeListKnownWithIds, studentIds = encodeListKnownWithIds
print(studentIds)
print("Encoded files Loaded")

modeType = 0
counter = 0
id = -1
imgStudent = []

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar cascade classifier
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Detect facial landmarks using dlib
            face_rect = dlib.rectangle(x, y, x + w, y + h)
            shape = predictor(gray, face_rect)
            shape = face_utils.shape_to_np(shape)

            # Calculate the center of the face using the detected facial landmarks
            face_center = np.mean(shape[28:30], axis=0).astype(int)

            # Calculate the distance between the face center and the bottom of the mask
            distance_to_mask = abs(face_center[1] - (y + h))

            if distance_to_mask < h / 2:
                # Partially hidden face detected

                # Encode the partially hidden face if there are face encodings available
                encodeFaceList = face_recognition.face_encodings(roi_color)
                if len(encodeFaceList) > 0:
                    encodeFace = encodeFaceList[0]

                    matches = face_recognition.compare_faces(encodeListKnownWithIds, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnownWithIds, encodeFace)

                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:

                        y1, x2, y2, x1 = y, x+w, y+h, x
                        bbox = 40 + x1, 145 + y1, x2 - x1, y2 - y1
                        imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                        id = studentIds[matchIndex]
                        if counter == 0:
                            cvzone.putTextRect(imgBackground, "Loading..", (275, 400))
                            cv2.imshow("Face Attendance", imgBackground)
                            cv2.waitKey(1)
                            counter = 1
                            modeType = 1

        if counter != 0:
            if counter == 1:
                # Get the data
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)

                # Get the image from the storage
                blob = bucket.get_blob(f'Images/{id}.jpeg')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                # Update attendance data
                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                print(secondsElapsed)

                if secondsElapsed > 30:
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total_attendance'] += 1
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                else:
                    modeType = 3
                    counter = 0
                    imgBackground[30:30 + 633, 825:825 + 414] = imgModeList[modeType]

            if modeType != 3:
                if 10 < counter < 20:
                    modeType = 2

                imgBackground[30:30 + 633, 825:825 + 414] = imgModeList[modeType]

                if counter <= 10:
                    cv2.putText(imgBackground, str(studentInfo['total_attendance']), (900, 105),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['major']), (1040, 525),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(id), (1006, 455),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['standings']), (950, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['year']), (1050, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['starting year']), (1150, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (410 - w) // 2


                    cv2.putText(imgBackground, str(studentInfo['name']), (825 + offset, 410),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                    imgBackground[150:150 + 216, 910:910 + 216] = imgStudent

                counter += 1

                if counter >= 30:
                    counter = 0
                    modeType = 0
                    studentInfo = []
                    imgStudent = []
                    imgBackground[30:30 + 633, 825:825 + 414] = imgModeList[modeType]
    else:
        modeType = 0
        counter = 0

    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)
