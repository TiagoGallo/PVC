import cv2
import numpy as np
import argparse

class Req1:
    def __init__(self, args):
        cv2.namedWindow("Requisito 1")
        cv2.setMouseCallback('Requisito 1',self.get_mouse_position)
        self.cor = (None, None, None)

        self.imagem = cv2.imread(args["imagem"])
        self.isGray = self.checkIfGray()

    def get_mouse_position(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("coluna = {}  linha = {}".format(x,y))
            
            #pega a cor no formato BGR
            self.cor = (self.imagem[y][x][0], self.imagem[y][x][1], self.imagem[y][x][2])

            if self.isGray:
                print("Intensidade do tom de cinza = {}".format(self.cor[0]))
            else:
                print("R = {}  G = {}  B = {}".format(self.cor[2], self.cor[1], self.cor[0]))

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

    def run(self):
        while True:
            cv2.imshow("Requisito 1", self.imagem)

            #press ESC to quit
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagem", default="cores.jpg",
        help="path to input image")
    args = vars(ap.parse_args())
    
    
    t = Req1(args)
    t.run()