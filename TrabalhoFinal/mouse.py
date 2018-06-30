import pyautogui
import time

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

        # Time counter to click the mouse
        self.timeStopped = time.time()
        self.clicked = False    # not a good solution

    def update(self, state):
        '''
        Analyze what to do with the mouse
        '''
        # get the mouse position
        w_pos, h_pos = self.actual_position()

        # call the function to update the accelarators
        self.move(state)

        # check if there is need to click
        self.rightClick(state)

        #print("[DEBUG] accX = {}\taccY = {}".format(self.accX, self.accY))

    def move(self, state):
        '''
        Receive the program state as [X, Y]
            X: -1 = go left    0 = nothing     1 = go right
            Y: -1 = go down     0 = nothing     1 = go up
        '''
        X, Y = state

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

    def rightClick(self, state):
        '''
        Check if we need to click
        '''
        # if there is no movement, check if is stopped for more time than the delay
        if state == [0,0]:
            elapsed_stop = time.time() - self.timeStopped
            if elapsed_stop > self.delay and not self.clicked:
                pyautogui.click()  # click the mouse
                self.clicked = True
        else: 
            # update the time counter
            self.timeStopped = time.time()
            self.clicked = False

        print("[DEBUG] tempo parado = ", time.time() - self.timeStopped)
