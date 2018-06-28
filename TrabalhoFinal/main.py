import cv2
import face_recognition
import argparse
import numpy as np

# landmark color mapping for drawing
LANDMARK_COLOR_MAP = {
    "left_eye": (255, 0, 0), #Desenha os pontos do olho esquerdo de azul
    "right_eye": (255, 0, 0), #Desenha os pontos do olho direito de azul
    "left_eyebrow": (0, 0, 255), #Desenha os pontos da sobrancelha esquerda de vermelho
    "right_eyebrow": (0, 0, 255), #Desenha os pontos da sobrancelha direita de vermelho
    "nose_tip": (0, 255, 0), #Desenha os pontos da ponta do nariz de verde
    "nose_bridge": (255, 255, 0), #Desenha os pontos do corpo do nariz de ciano
    "bottom_lip": (0, 255, 255), #Desenha os pontos do labio de amarelo
    "top_lip": (0, 255, 255), #Desenha os pontos do labio de amarelo
    "chin": (0, 0, 0) #Desenha os pontos da bochecha de preto
}


def draw_landmarks(img, landmarks, landmark_color_map):
    """
    Draws the detected face landmarks on the image according to the given color map.
    """
    for landmark, color in landmark_color_map.items():
        for point in landmarks[landmark]:
            cv2.circle(img, point, 1, color, -1)


def draw_bbox(img, location):
    """
    Draws a bounding box on the image for each face on the list of given face locations.
    """
    (top, right, bottom, left) = location
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))


def largest_face(face_locations):
    """
    Determine the largest face detected and return its location.
    """
    max_size = 0
    max_location = None
    for (top, right, bottom, left) in face_locations:
        size = (bottom - top) * (right - left)
        if size > max_size:
            max_size = size
            max_location = (top, right, bottom, left)
    return max_location


def user_state(img, landmarks):
    """
    Determine the user state from the detected landmarks.
    """
    # thresholds for movement
    x_lim_left = -10
    x_lim_right = 10
    y_lim_top = 30
    y_lim_bottom = 10

    # pixel difference in x
    nose_top = landmarks['nose_bridge'][0][0]
    nose_bottom = landmarks['nose_bridge'][3][0]
    x_diff = nose_bottom - nose_top

    # pixel difference in y
    eyebrows_center = (landmarks['left_eyebrow'][4][1] + landmarks['right_eyebrow'][0][1]) // 2
    eyes_center = (landmarks['left_eye'][3][1] + landmarks['right_eye'][0][1]) // 2
    y_diff_eye = eyes_center - eyebrows_center
    nose_top = landmarks['nose_bridge'][0][1]
    nose_bottom = landmarks['nose_bridge'][3][1]
    y_diff_nose = nose_bottom - nose_top

    # determine if there should be movement
    print('[DEBUG] pixel diff: ({}, {}/{})'.format(x_diff, y_diff_eye, y_diff_nose))
    state = [0, 0]
    if x_diff < -10:
        state[0] = -1
    elif x_diff > 10:
        state[0] = 1
    if y_diff_eye < 10:
        state[1] = -1
    elif y_diff_nose < 40:
        state[1] = 1
    return state

def main():
    """
    Main function.
    """

    # setup argument parser
    ap = argparse.ArgumentParser('Settings for the mouse.')

    # mode arguments
    ap.add_argument('-d', '--delay', type=int, default=1,
        help='Delay before registering mouse click.')
    # ap.add_argument('-r', '--recovery', type=int, default=-1,
    #     help='Recovery time (minimum waiting time) between clicks.')
    # ap.add_argument('-m', '--movement', type=str, default='rel',
    #     help='Mouse movement mode. Can be \'rel\' (default) or \'abs\'.')

    # parse the arguments
    args = vars(ap.parse_args())

    # create the capture object with the default webcam
    cam = cv2.VideoCapture(0)

    # counter for skipping frames
    counter = 0
    numSkippedFrames = 2

    while True:

        # try to read the frame
        grabbed, img = cam.read()
        if img is None or grabbed is False:
            break

        # skip frames when the counter is not 0
        if counter == 0:

            # try to find faces on the image
            face_locations = face_recognition.face_locations(img, model="hog")

            # if faces were found, process them
            if (len(face_locations) > 0):

                # if there is more than one face on the image, pick the largest one
                if (len(face_locations) > 1):
                    location = largest_face(face_locations)
                else:
                    location = face_locations[0]

                # find the facial landmarks on the face
                landmarks = face_recognition.face_landmarks(img, face_locations=[location])[0]

                if (len(landmarks) > 0):
                    # get the relative pose of the face
                    state = user_state(img, landmarks)

                    # move mouse according to state
                    pass

                    # draw the bounding box and the landmarks on the frame
                    draw_bbox(img, location)
                    draw_landmarks(img, landmarks, LANDMARK_COLOR_MAP)

            # show the frame
            cv2.imshow("Webcam", img)
        
        counter = (counter + 1) % numSkippedFrames

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def calc_relative_nose(face_locations, face_landmarks_list):
    for face_location, landmark in zip(face_locations, face_landmarks_list):
        point_med_nose = [0,0]
        for i, point in enumerate(landmark["nose_tip"]):
            point_med_nose[0] = point_med_nose[0] + point[0]
            point_med_nose[1] = point_med_nose[1] + point[1]

        point_med_nose[0] = point_med_nose[0] // i
        point_med_nose[1] = point_med_nose[1] // i
        print("[DEBUG] Ponto medio do nariz = ", point_med_nose)


if __name__ == "__main__":
    main()