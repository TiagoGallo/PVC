import cv2
import face_recognition

def find_main_face(img, model):
    """
    Try to find the main face on the image (the largest one).
    """
    # recognize faces on the image
    face_locations = face_recognition.face_locations(img, model=model)

    # if no faces were found, indicate the failure
    if len(face_locations) == 0:
        return None

    # if there is more than one face on the image, pick the largest one
    elif len(face_locations) > 1:
        max_size = 0
        max_location = None
        for (top, right, bottom, left) in face_locations:
            size = (bottom - top) * (right - left)
            if size > max_size:
                max_size = size
                max_location = (top, right, bottom, left)
        return max_location

    # if there is only one, return the list element
    else:
        return face_locations[0]

def find_landmarks(img, location):
    """
    Try to find the landmarks on the detected face.
    """
    landmarks = face_recognition.face_landmarks(img, face_locations=[location])
    if len(landmarks) == 0:
        return None
    else:
        return landmarks[0]