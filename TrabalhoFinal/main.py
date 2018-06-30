# import packages
import cv2
import numpy as np
import argparse
import draw
import detect
from mouse import Mouse
from state import State

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

def main():
    """
    Main function.
    """

    # setup argument parser
    ap = argparse.ArgumentParser('Settings for the mouse.')
    ap.add_argument('-d', '--delay', type=int, default=3,
        help='Delay before registering mouse click.')
    ap.add_argument('-c', '--click-mode', type=str, default='dwell',
        help='Clicking method. Can be \'dwell\' or \'eye\'.')
    ap.add_argument('-m', '--max_acc', type=int, default=20,
        help='Velocity of the mouse')
    args = vars(ap.parse_args())

    # Create OpenCV window and trackbars (and initialize them to default values)
    cv2.namedWindow("Webcam")
    cv2.createTrackbar('Up', 'Webcam', 50, 100, lambda: None)
    cv2.createTrackbar('Down', 'Webcam', 50, 100, lambda: None)
    cv2.createTrackbar('Left', 'Webcam', 50, 100, lambda: None)
    cv2.createTrackbar('Right', 'Webcam', 50, 100, lambda: None)
    cv2.createTrackbar('Sens', 'Webcam', 100, 200, lambda: None)
    if args['click_mode'] == 'eye':
        cv2.createTrackbar('Eye AR', 'Webcam', 10, 100, lambda: None)

    # create video capture object with the default webcam
    cam = cv2.VideoCapture(0)

    # create the user state object
    state = State()

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

            # try to find the main face on the image
            location = detect.find_main_face(img, model="hog")
            if location is not None:
            
                # try to find the landmarks on the detected face
                landmarks = detect.find_landmarks(img, location)
                if landmarks is not None:

                    # get the relative pose of the face
                    state.update_state(img, landmarks, location)

                    # draw the bounding box and the landmarks on the frame
                    draw.draw_bbox(img, location)
                    draw.draw_landmarks(img, landmarks, LANDMARK_COLOR_MAP)

                    # move mouse according to state
                    mouse.update(state)

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