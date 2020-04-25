import face_recognition
import cv2
import numpy as np

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)


def isRaise(face_landmark):
    for face in face_landmark:
        a = (face['right_eyebrow'][0][0] - face['left_eyebrow'][0][0]) / (
                    face['top_lip'][0][1] - face['nose_bridge'][0][1])
        return 'down'


def raiseDetection(frame, scale):
    # Initialize some variables
    face_locations = []
    process_this_frame = True

    # Grab a single frame of video

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=1 / scale, fy=1 / scale)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
    process_this_frame = not process_this_frame

    # Display the results
    raiseNum = 0

    for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a namie below the face
        cv2.rectangle(frame, (left, bottom - (top - bottom) // 5), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX

        position = isRaise(face_recognition.face_landmarks(frame))
        cv2.putText(frame, position, (left + 6, bottom - (top - bottom) // 5 - 6), font, 1.0, (255, 255, 255), 1)
        if position == 'raise':
            raiseNum += 1
    return frame, face_locations, raiseNum
