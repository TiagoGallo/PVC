import cv2
import numpy as np

class State:
    def __init__(self, mov=None, eye=None):
        self.mov = mov
        self.eye = eye

    def eye_aspect_ratio(self, eye):
        """
        Compute the eye aspect ratio.
        """
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = np.linalg.norm(np.array(eye[1])-np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2])-np.array(eye[4]))
    
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = np.linalg.norm(np.array(eye[0])-np.array(eye[3]))
    
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
    
        # return the eye aspect ratio
        return ear

    def update_state(self, img, landmarks, location):
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
        ear_thresh = cv2.getTrackbarPos('Eye AR','Webcam')/100

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
        self.mov = [0, 0]
        #Movimento para esquerda
        if focus_x < x_lim_left:
            self.mov[0] = -1
        #Movimento para direita
        elif focus_x > x_lim_right:
            self.mov[0] = 1
        #movimento para baixo
        if focus_y > y_lim_bottom:
            self.mov[1] = -1
        #Movimento para cima
        elif focus_y < y_lim_top:
            self.mov[1] = 1

        # determine eye state
        self.eye = [0, 0]
        ear_left = self.eye_aspect_ratio(landmarks['left_eye'])
        ear_right = self.eye_aspect_ratio(landmarks['right_eye'])
        if ear_left < ear_thresh and ear_right < ear_thresh:
            self.eye = [1, 1]
        elif ear_left < ear_thresh or ear_right < ear_thresh:
            if ear_left < ear_right:
                self.eye[0] = 1
            else:
                self.eye[1] = 1