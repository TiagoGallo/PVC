import cv2
import state
import numpy as np

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

def drawArrow(state, img):
    '''
    Draw an arrow that indicate the mouse movement
    '''
    X, Y = state.mov

    if X == 0 and Y == 0:
        return
    elif X == 1 and Y == 0:
        cv2.arrowedLine(img, (20,50), (100, 50), (0,0,255), 3)
    elif X == -1 and Y == 0:
        cv2.arrowedLine(img, (100, 50), (20,50), (0,0,255), 3)
    elif X == 0 and Y == 1:
        cv2.arrowedLine(img, (60, 100), (60,20), (0,0,255), 3)
    elif X == 1 and Y == 1:
        cv2.arrowedLine(img, (20, 100), (100,20), (0,0,255), 3)
    elif X == -1 and Y == 1:
        cv2.arrowedLine(img, (100, 100), (20,20), (0,0,255), 3)
    elif X == 0 and Y == -1:
        cv2.arrowedLine(img, (60, 20), (60, 100), (0,0,255), 3)
    elif X == 1 and Y == -1:
        cv2.arrowedLine(img, (20, 20), (100,100), (0,0,255), 3)
    elif X == -1 and Y == -1:
        cv2.arrowedLine(img, (100, 20), (20,100), (0,0,255), 3)

def draw_eyes(img, state, eye_open, eye_closed):
    """
    Draws the eyes state on the image.
    """
    # info about the image and the eyes
    esize = int(0.1 * img.shape[1])
    left = int(0.775 * img.shape[1])
    right = int(0.875 * img.shape[1])
    height = int(0.05 * img.shape[0])

    # eyes image and mask
    eyes_img = np.zeros_like(img)
    eyes_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # left eye
    if state.eye[0] == 0:
        eyes_img[height:height+esize, left:left+esize, :] = eye_open[:, :, :]
    elif state.eye[0] == 1:
        eyes_img[height:height+esize, left:left+esize, :] = eye_closed[:, :, :]

    # right eye
    if state.eye[1] == 0:
        eyes_img[height:height+esize, right:right+esize, :] = eye_open[:, :, :]
    elif state.eye[1] == 1:
        eyes_img[height:height+esize, right:right+esize, :] = eye_closed[:, :, :]

    # join the two images
    eyes_mask[eyes_img[:, :, 0] == 255] = 255
    background_mask = 255 - eyes_mask
    masked_img = cv2.bitwise_and(img, img, mask=background_mask)
    masked_eyes = cv2.bitwise_and(eyes_img, eyes_img, mask=eyes_mask)
    cv2.bitwise_or(masked_img, masked_eyes, dst=img)