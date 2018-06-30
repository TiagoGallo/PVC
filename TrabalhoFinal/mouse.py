import pyautogui

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
            X: -1 = go left    0 = nothing     1 = go right
            Y: -1 = go down     0 = nothing     1 = go up
        '''
        X, Y = state
        
        w_pos, h_pos = self.actual_position()

        # Analyze the x movement
        if X == 0:
            self.att_acc('X')    #decrease the acc if different of 0
        elif X == 1:
            self.accX += 1 # Move right
            if self.accX > self.accMax: self.accX = self.accMax # Limit the moviment
        elif X == -1:
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