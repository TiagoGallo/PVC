import cv2
import face_recognition
import argparse
import numpy as np
import pyautogui

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

def nothing(x):
    pass

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
    x_lim_right = -10 + cv2.getTrackbarPos('Right','Webcam')
    x_lim_left = 10 - cv2.getTrackbarPos('Left','Webcam')
    y_lim_top = 30 + cv2.getTrackbarPos('Up','Webcam')
    y_lim_bottom = 10 + cv2.getTrackbarPos('Down','Webcam')

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
    #Movimento para direita
    if x_diff < x_lim_right:
        print("[DEBUG] Direita")
        state[0] = -1
    #Movimento para esquerda
    elif x_diff > x_lim_left:
        print("[DEBUG] Esquerda")
        state[0] = 1
    #movimento para baixo
    if y_diff_eye < y_lim_bottom:
        print("[DEBUG] Baixo")
        state[1] = -1
    #Movimento para cima
    elif y_diff_nose < y_lim_top:
        print("[DEBUG] Cima")
        state[1] = 1
    return state

class Mouse:
    def __init__(self, args):
        self.width, self.height = pyautogui.size()
        print("[DEBUG] O tamanho da janela eh {}x{}".format(self.width, self.height))

        # Create an accelarator for both directions and a max value for them
        self.accX   = 0
        self.accY   = 0
        self.accMax = 10

        # Delay to click the mouse
        self.delay = args["delay"]

    def move(self, state):
        '''
        Receive the program state as [X, Y]
            X: -1 = go right    0 = nothing     1 = go left
            Y: -1 = go down     0 = nothing     1 = go up
        '''
        X, Y = state
        
        w_pos, h_pos = self.actual_position()

        # Analyze the x movement
        if X == 0:
            self.att_acc('X')    #decrease the acc if different of 0
        elif X == -1:
            self.accX += 1 # Move right
            if self.accX > self.accMax: self.accX = self.accMax # Limit the moviment
        elif X == 1:
            self.accX -= 1 # Move left
            if self.accX < -(self.accMax): self.accX = -(self.accMax) # Limit the moviment

        # Analyze the y movement
        if Y == 0:
            self.att_acc('Y')    #decrease the acc if different of 0
        elif Y == -1:
            self.accY += 1 # Move down
            if self.accY > self.accMax: self.accY = self.accMax # Limit the moviment
        elif Y == 1:
            self.accY -= 1 # Move up
            if self.accY < -(self.accMax): self.accY = -(self.accMax) # Limit the moviment

        pyautogui.moveRel(self.accX, self.accY)

    def actual_position(self):
        '''
        Get mouse's actual position
        '''
        w_pos, h_pos = pyautogui.position()

        return [w_pos, h_pos]

    def att_acc(self, axis):
        '''
        Method to desaccelerate the accelerator
        '''
        if axis == 'X':
            if self.accX > 0:
                self.accX -= 1
            elif self.accX < 0:
                self.accX += 1

        if axis == 'Y':
            if self.accY > 0:
                self.accY -= 1
            elif self.accY < 0:
                self.accY += 1

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

    # Create OpenCV window and trackbars
    cv2.namedWindow("Webcam")
    cv2.createTrackbar('Up','Webcam',0,255,nothing)
    cv2.createTrackbar('Down','Webcam',0,255,nothing)
    cv2.createTrackbar('Left','Webcam',0,255,nothing)
    cv2.createTrackbar('Right','Webcam',0,255,nothing)

    # create the capture object with the default webcam
    cam = cv2.VideoCapture(0)

    # create the mouse object
    mouse = Mouse(args)

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
                    mouse.move(state)

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