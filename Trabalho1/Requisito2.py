import cv2
import numpy as np

# mouse callback function
def get_mouse_position(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("coluna = {}  linha = {}".format(x,y))
        
        #pega a cor no formato BGR
        cor = (imagem[y][x][0], imagem[y][x][1], imagem[y][x][2])
        
        print("R = {}  G = {}  B = {}".format(cor[2], cor[1], cor[0]))
        
        paint_pixels(x, y, cor)

def paint_pixels(x, y, cor):
    #pega as caracteristicas da imagem e percorre ela
    width, height, _ = imagem.shape

    for w in np.arange(0, width):
        for h in np.arange(0, height):
            color = (imagem[w][h][0], imagem[w][h][1], imagem[w][h][2])
            if is_same_color(cor, color):
                imagem[w][h] = (0, 0, 255)

def is_same_color(cor, color):
    dB = (int(cor[0]) - int(color[0])) ** 2
    dG = (int(cor[1]) - int(color[1])) ** 2
    dR = (int(cor[2]) - int(color[2])) ** 2

    dist = (dB + dG + dR) ** (0.5)

    if dist < 13:
        return True
    else:
        return False

imagem = cv2.imread("cores.png")

cv2.namedWindow("Requisito 2")
cv2.setMouseCallback('Requisito 2',get_mouse_position)

while True:
    cv2.imshow("Requisito 2", imagem)
   
   #press ESC to quit
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
