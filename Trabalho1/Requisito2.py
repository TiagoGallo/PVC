import cv2
import numpy as np
import argparse

class Req2:
    def __init__(self, args):
        cv2.namedWindow("Requisito 2")
        cv2.setMouseCallback('Requisito 2',self.get_mouse_position)
        self.cor = (None, None, None)

        self.imagemOriginal = cv2.imread(args["imagem"])
        self.imagem = self.imagemOriginal.copy()
        self.isGray = self.checkIfGray()

    def get_mouse_position(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("coluna = {}  linha = {}".format(x,y))

            #pega a cor no formato BGR
            self.cor = (self.imagem[y][x][0], self.imagem[y][x][1], self.imagem[y][x][2])

            self.imagem = self.imagemOriginal.copy()

            if self.isGray:
                print("Intensidade do tom de cinza = {}".format(self.cor[0]))
            else:
                print("R = {}  G = {}  B = {}".format(self.cor[2], self.cor[1], self.cor[0]))

            self.cor = (self.imagem[y][x][0], self.imagem[y][x][1], self.imagem[y][x][2])

    def checkIfGray(self):
        #pega as caracteristicas da self.imagem e percorre ela
        width, height, depth = self.imagem.shape

        if depth == 1:
            return True
            
        for w in np.arange(0, width):
            for h in np.arange(0, height):
                if not ((self.imagem[w][h][0] == self.imagem[w][h][1]) and (self.imagem[w][h][1] == self.imagem[w][h][2])):
                    return False

        return True

    def paint_pixels(self):
        #pega as caracteristicas da imagem e percorre ela
        width, height, _ = self.imagemOriginal.shape

        if self.cor == (None, None, None):
            return

        for w in np.arange(0, width):
            for h in np.arange(0, height):
                color = (self.imagemOriginal[w][h][0], self.imagemOriginal[w][h][1], self.imagemOriginal[w][h][2])
                if self.is_same_color(self.cor, color):
                    self.imagem[w][h] = (0, 0, 255)

    def is_same_color(self, cor, color):
        dB = (int(cor[0]) - int(color[0])) ** 2
        dG = (int(cor[1]) - int(color[1])) ** 2
        dR = (int(cor[2]) - int(color[2])) ** 2

        dist = (dB + dG + dR) ** (0.5)

        if dist < 13:
            return True
        else:
            return False

    def run(self):
        while True:
            
            self.paint_pixels()
            
            cv2.imshow("Requisito 2", self.imagem)

            #press ESC to quit
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagem", default="cores.jpg",
        help="path to input image")
    args = vars(ap.parse_args())
    
    
    t = Req2(args)
    t.run()
