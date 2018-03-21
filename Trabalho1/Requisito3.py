import cv2
import numpy as np

class Teste:
    def __init__(self):
        self.vid = cv2.VideoCapture("SampleVideo.mp4")
        cv2.namedWindow("Requisito 3")
        cv2.setMouseCallback('Requisito 3',self.get_mouse_position)
        self.cor = (None, None, None)

    def get_mouse_position(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("coluna = {}  linha = {}".format(x,y))
            
            #pega a cor no formato BGR
            self.cor = (self.imagem[y][x][0], self.imagem[y][x][1], self.imagem[y][x][2])
            
            print("R = {}  G = {}  B = {}".format(self.cor[2], self.cor[1], self.cor[0]))

    def paint_pixels(self):
        #pega as caracteristicas da self.imagem e percorre ela
        width, height, _ = self.imagem.shape

        if self.cor == (None, None, None):
            return
            
        for w in np.arange(0, width):
            for h in np.arange(0, height):
                color = (self.imagem[w][h][0], self.imagem[w][h][1], self.imagem[w][h][2])
                if self.is_same_color(color):
                    self.imagem[w][h] = (0, 0, 255)

    def is_same_color(self, color):
        dB = (int(self.cor[0]) - int(color[0]))
        if dB >= 13: return False

        dG = (int(self.cor[1]) - int(color[1]))
        if dG >= 13: return False

        dR = (int(self.cor[2]) - int(color[2]))
        if dR >= 13: return False

        if ((dB)**2 + (dG)**2 + (dR)**2) >= 169: return False

        return True
       

    def run(self):
        while True:
            grabbed, self.imagem = self.vid.read()

            if not grabbed or self.imagem is None:
                break
        
            self.paint_pixels()

            cv2.imshow("Requisito 3", self.imagem)

            #press ESC to quit
            if self.cor == (None, None, None):
                k = cv2.waitKey(30) & 0xFF
            else:
                k = cv2.waitKey(1) & 0xFF
            
            if k == ord("q"):
                break

if __name__ == "__main__":
    t = Teste()
    t.run()