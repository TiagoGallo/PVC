import pyautogui
import time
import cv2
import numpy as np
import draw
from state import State

class Mouse:
    def __init__(self, args):
        self.width, self.height = pyautogui.size()
        #print("[DEBUG] O tamanho da janela eh {}x{}".format(self.width, self.height))

        # Create an accelarator for both directions and a max value for them
        self.accX   = 0
        self.accY   = 0
        self.accMax = args["max_acc"]

        # Delay to click the mouse
        self.delay = args["delay"]
        self.mode = args['click_mode']

        # Time counter to click the mouse
        self.timeStopped = time.time()
        self.clicked = False    # not a good solution

        # time counters for the eyes
        self.timeLeftEyeClosed = time.time()
        self.timeRightEyeClosed = time.time()
        self.timeBothEyesClosed = time.time()

        # loads the eye icons and color them blue
        eye_open_raw = cv2.imread('eye-open.png', cv2.IMREAD_UNCHANGED)
        eye_closed_raw = cv2.imread('eye-closed.png', cv2.IMREAD_UNCHANGED)
        self.eye_open = np.zeros((eye_open_raw.shape[0], eye_open_raw.shape[1], 3))
        self.eye_closed = np.zeros((eye_closed_raw.shape[0], eye_closed_raw.shape[1], 3))
        self.eye_open[eye_open_raw[:, :, 3] == 255] = [255, 0, 0]
        self.eye_closed[eye_closed_raw[:, :, 3] == 255] = [255, 0, 0]    

    def update(self, state, img):
        '''
        Analyze what to do with the mouse
        '''
        # get the mouse position
        w_pos, h_pos = self.actual_position()

        # draw the eyes according to the state
        eye_size = int(0.1*img.shape[1])
        if self.eye_open.shape != (eye_size, eye_size, 4):
            self.eye_open = cv2.resize(self.eye_open, (eye_size, eye_size))
        if self.eye_closed.shape != (eye_size, eye_size, 4):
            self.eye_closed = cv2.resize(self.eye_closed, (eye_size, eye_size))
        draw.draw_eyes(img, state, self.eye_open, self.eye_closed)

        # call the function to update the accelarators
        self.move(state)

        # check if there is need to click
        if self.mode == 'dwell':
            self.dwell_click(state)
        elif self.mode == 'eye':
            self.eye_click(state)
        elif self.mode == 'simple-eye':
            self.simple_eye_click(state)

        draw.drawArrow(state, img)
        #print("[DEBUG] accX = {}\taccY = {}".format(self.accX, self.accY))

    def move(self, state):
        '''
        Receive the program state as [X, Y]
            X: -1 = go left    0 = nothing     1 = go right
            Y: -1 = go down     0 = nothing     1 = go up
        '''
        X, Y = state.mov

        # Analyze the x movement
        if X == 0:
            self.att_acc('X')    #decrease the acc if different of 0
        elif X == 1:
            self.accX += self.accMax // 10 # Move right
            if self.accX > self.accMax: self.accX = self.accMax # Limit the moviment
        elif X == -1:
            self.accX -= self.accMax // 10 # Move left
            if self.accX < -(self.accMax): self.accX = -(self.accMax) # Limit the moviment

        # Analyze the y movement
        if Y == 0:
            self.att_acc('Y')    #decrease the acc if different of 0
        elif Y == -1:
            self.accY += self.accMax // 10 # Move down
            if self.accY > self.accMax: self.accY = self.accMax # Limit the moviment
        elif Y == 1:
            self.accY -= self.accMax // 10 # Move up
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
            if self.accX == -1:
                self.accX = 0 
            else: 
                self.accX = self.accX // 2

        if axis == 'Y':
            if self.accY == -1:
                self.accY = 0 
            else: 
                self.accY = self.accY // 2

    def dwell_click(self, state):
        '''
        Check if we need to click (dwell mode)
        '''
        # if there is no movement, check if is stopped for more time than the delay
        if state.mov == [0,0]:
            elapsed_stop = time.time() - self.timeStopped
            if elapsed_stop > self.delay and not self.clicked:
                pyautogui.click()  # click the mouse
                self.clicked = True
        else: 
            # update the time counter
            self.timeStopped = time.time()
            self.clicked = False

        print("[DEBUG] tempo parado = ", time.time() - self.timeStopped)

    def eye_click(self, state):
        '''
        Check if we need to click (eye mode)
        '''
        # the left eye is closed
        if state.eye == [1, 0]:

            # update the timers on the right eye and on both eyes
            self.timeRightEyeClosed = time.time()
            self.timeBothEyesClosed = time.time()

            # check for how much time the left eye has been closed
            elapsed_stop = time.time() - self.timeLeftEyeClosed
            if elapsed_stop > self.delay and not self.clicked:
                pyautogui.click()
                self.clicked = True

            print("[DEBUG] tempo com olho esquerdo fechado = ", elapsed_stop)
        
        # the right eye is closed
        elif state.eye == [0, 1]:

            # update the timers on the left eye and on both eyes
            self.timeLeftEyeClosed = time.time()
            self.timeBothEyesClosed = time.time()

            # check for how much time the right eye has been closed
            elapsed_stop = time.time() - self.timeRightEyeClosed
            if elapsed_stop > self.delay and not self.clicked:
                pyautogui.rightClick()
                self.clicked = True

            print("[DEBUG] tempo com olho direito fechado = ", elapsed_stop)

        # both eyes are closed
        elif state.eye == [1, 1]:

            # update the timers on the left eye and on the right eye
            self.timeLeftEyeClosed = time.time()
            self.timeRightEyeClosed = time.time()

            # check for how much time both eyes has been closed
            elapsed_stop = time.time() - self.timeBothEyesClosed
            if elapsed_stop > self.delay and not self.clicked:
                pyautogui.doubleClick()
                self.clicked = True

            print("[DEBUG] tempo com os dois olhos fechado = ", elapsed_stop)

        # both eyes are open
        if state.eye == [0, 0]:
            # update all time counters
            self.timeLeftEyeClosed = time.time()
            self.timeRightEyeClosed = time.time()
            self.timeBothEyesClosed = time.time()
            self.clicked = False

    def simple_eye_click(self, state):
        '''
        Check if we need to click (simple eye mode)
        '''
        # both eyes are closed
        if state.eye == [1, 1]:

            # check for how much time both eyes has been closed
            elapsed_stop = time.time() - self.timeBothEyesClosed
            if elapsed_stop > self.delay and not self.clicked:
                pyautogui.click()
                self.clicked = True

            print("[DEBUG] tempo com os dois olhos fechado = ", elapsed_stop)

        # update time counter
        else:
            self.timeBothEyesClosed = time.time()
            self.clicked = False