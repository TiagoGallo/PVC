import cv2
import numpy as np
import argparse

class Req1:
    def run(self, args):
        self.readImages(args)

        self.matrixOfCorrespondences = np.zeros((self.imgLeft.shape[0], self.imgLeft.shape[1]), dtype=tuple)

        self.create_kernel_and_padImage(args)
        self.calcL1Norm()

        #Agrupa as duas imagens em uma só e mostra o resultado
        self.resize_and_stack()
        cv2.imshow("Both", self.imageFinal)
        cv2.waitKey(0)

    def readImages(self, args):
        #Lê as duas imagens
        if args["image"] == '0':
            self.imgRight = cv2.imread('./imgs-estereo/aloeL.png') 
            self.imgLeft = cv2.imread('./imgs-estereo/aloeR.png') 
        
        elif args["image"] == '1':
            self.imgRight = cv2.imread('./imgs-estereo/babyL.png') 
            self.imgLeft = cv2.imread('./imgs-estereo/babyR.png')
        
        else:
            raise NameError ("O argumento -i so aceita os parametros 0 ou 1")

    def create_kernel_and_padImage(self, args):
        if int(args["size"]) % 2 == 0:
            raise NameError ("O argumento -s so aceita numeros impares")

        self.kernel = np.zeros((int(args["size"]), int(args["size"])), dtype=int)    

        self.borderSize = int(args["size"]) // 2

        self.imgLeft = cv2.copyMakeBorder(self.imgLeft,self.borderSize,self.borderSize,self.borderSize,self.borderSize,cv2.BORDER_CONSTANT,value=0)
        self.imgRight = cv2.copyMakeBorder(self.imgRight,self.borderSize,self.borderSize,self.borderSize,self.borderSize,cv2.BORDER_CONSTANT,value=0)

    def calcL1Norm(self):
        h, w = self.imgLeft.shape

        for i in range(self.borderSize, h - self.borderSize):
            for j in range(self.borderSize, w - self.borderSize):
                

    def resize_and_stack (self):
        size = self.imgLeft.shape

        r = 3
        h_new = int(size[0] / r)
        w_new = int(size[1] / r)
        self.imgLeft = cv2.resize(self.imgLeft, (h_new, w_new))
        self.imgRight = cv2.resize(self.imgRight, (h_new, w_new))

        self.imageFinal = np.hstack((self.imgLeft, self.imgRight))


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--size", default='3',
        help="Qual o tamanho da janela para calcular o L1 norm?")
    ap.add_argument("-i", "--image", default='0',
        help="Qual conjunto de imagens deseja usar? 0 = planta e 1 = baby")
    args = vars(ap.parse_args())
    
    Req1().run(args)