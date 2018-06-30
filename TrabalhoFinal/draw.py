import cv2

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