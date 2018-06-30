import cv2
import face_recognition
import argparse
import numpy as np
import pyautogui
from mouse import Mouse

# landmark color mapping for drawing
LANDMARK_COLOR_MAP = {
    "left_eye": (255, 0, 0), #Desenha os pontos do olho esquerdo de azul
    "right_eye": (255, 0, 0), #Desenha os pontos do olho direito de azul
    "left_eyebrow": (0, 0, 255), #Desenha os pontos da sobrancelha esquerda de vermelho
    "right_eyebrow": (0, 0, 255), #Desenha os pontos da sobrancelha direita de vermelho
    "nose_tip": (0, 255, 0), #Desenha os pontos da ponta do nariz de verde
    "nose_bridge": (255, 255, 0), #Desenha os pontos do corpo do nariz de ciano
    "bottom_lip": (0, 127, 255), #Desenha os pontos do labio de amarelo
    "top_lip": (0, 255, 127), #Desenha os pontos do labio de amarelo
    "chin": (0, 0, 0) #Desenha os pontos da bochecha de preto
}

def nothing(arg = None):
    pass

def create_trackbars():
    """
    Create the trackbars on the window and initialize them to the default values.
    """
    cv2.createTrackbar('Up', 'Webcam', 50, 100, nothing)
    cv2.createTrackbar('Down', 'Webcam', 50, 100, nothing)
    cv2.createTrackbar('Left', 'Webcam', 50, 100, nothing)
    cv2.createTrackbar('Right', 'Webcam', 50, 100, nothing)
    cv2.createTrackbar('Sens', 'Webcam', 100, 200, nothing)

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

def user_state(img, landmarks, location):
    """
    Determine the user state from the detected landmarks using camera geometry.
    """
    # thresholds for movement
    h, w = img.shape[0:2]
    x_lim_right = int(w/2 * (1 + (cv2.getTrackbarPos('Right','Webcam'))/100))
    x_lim_left = int(w/2 * (cv2.getTrackbarPos('Left','Webcam')/100))
    y_lim_top = int(h/2 * (cv2.getTrackbarPos('Up','Webcam')/100))
    y_lim_bottom = int(h/2 * (1 + (cv2.getTrackbarPos('Down','Webcam'))/100))
    sens = (cv2.getTrackbarPos('Sens','Webcam'))/100

    # draw the boundaries
    cv2.line(img, (x_lim_right, 0), (x_lim_right, h), (255, 0, 0))
    cv2.line(img, (x_lim_left, 0), (x_lim_left, h), (255, 0, 0))
    cv2.line(img, (0, y_lim_top), (w, y_lim_top), (255, 0, 0))
    cv2.line(img, (0, y_lim_bottom), (w, y_lim_bottom), (255, 0, 0))
        
    # 2D image points for the detected face
    image_points = np.array([
                                landmarks['nose_bridge'][3], # Nose tip
                                landmarks['chin'][8],        # Chin
                                landmarks['left_eye'][0],    # Left eye left corner
                                landmarks['right_eye'][3],   # Right eye right corne
                                landmarks['top_lip'][0],     # Left Mouth corner
                                landmarks['top_lip'][6]      # Right mouth corner
                            ], dtype=np.float)
    
    # 3D model points for a face in an arbitrary world frame
    model_points = np.array([
                                (0.0, 0.0, 0.0),          # Nose tip
                                (0.0, -66.0, -13.0),      # Chin
                                (-45.0, 34.0, -27.0),     # Left eye left corner
                                (45.0, 34.0, -27.0),      # Right eye right corne
                                (-30.0, -30.0, -25.0),    # Left Mouth corner
                                (30.0, -30.0, -25.0)      # Right mouth corner
                            ])
    
    # internal parameters for the camera (approximated)
    f = img.shape[1]
    c_x, c_y = (img.shape[1]/2, img.shape[0]/2)
    mtx = np.array([[f, 0, c_x],
                    [0, f, c_y],
                    [0, 0, 1]], dtype=np.float)
    dist = np.zeros((4,1))
    (ret, rvec, tvec) = cv2.solvePnP(model_points, image_points, mtx, dist)
    
    # project a 3D point (defined by the sensibility) onto the image plane
    # this is used to draw a line sticking out of the nose
    nose_end_3D = np.array([(0.0, 0.0, 100.0*sens)])
    (nose_end_2D, _) = cv2.projectPoints(nose_end_3D, rvec, tvec, mtx, dist)
    focus_x = int(nose_end_2D[0][0][0])
    focus_y = int(nose_end_2D[0][0][1])
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (focus_x, focus_y)
    cv2.line(img, p1, p2, (255,0,0), 2)
    for p in image_points:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    # determine if there should be movement
    state = [0, 0]
    #Movimento para esquerda
    if focus_x < x_lim_left:
        #print("[DEBUG] Esquerda")
        state[0] = -1
    #Movimento para direita
    elif focus_x > x_lim_right:
        #print("[DEBUG] Direita")
        state[0] = 1
    #movimento para baixo
    if focus_y > y_lim_bottom:
        #print("[DEBUG] Baixo")
        state[1] = -1
    #Movimento para cima
    elif focus_y < y_lim_top:
        #print("[DEBUG] Cima")
        state[1] = 1

    return state

def main():
    """
    Main function.
    """

    # setup argument parser
    ap = argparse.ArgumentParser('Settings for the mouse.')
    ap.add_argument('-d', '--delay', type=int, default=3,
        help='Delay before registering mouse click.')
    ap.add_argument('-m', '--max_acc', type=int, default=20,
        help='Velocity of the mouse')
    args = vars(ap.parse_args())

    # Create OpenCV window and trackbars
    cv2.namedWindow("Webcam")
    create_trackbars()

    # create the capture object with the default webcam
    cam = cv2.VideoCapture(0)

    # create the mouse object
    mouse = Mouse(args)

    # counter for skipping frames
    counter = 0
    numSkippedFrames = 2

    while True:

        # try to read the frame
        grabbed, frame = cam.read()
        if frame is None or grabbed is False:
            break
        
        # mirror the image horizontally (so that the processing is more intuitive)
        img = cv2.flip(frame, 1)

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
                    state = user_state(img, landmarks, location)

                    # move mouse according to state
                    mouse.update(state)

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

if __name__ == "__main__":
    main()