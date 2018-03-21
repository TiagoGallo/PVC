import cv2

# mouse callback function
def get_mouse_position(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONCLK:
        print("x = {}  y = {}".format(x,y))
        print("R = {}  G = {}  B = {}".format(imagem[y][x][2], imagem[y][x][1], imagem[y][x][0]))

imagem = cv2.imread("cores.png")

cv2.namedWindow("Requisito 1")
cv2.setMouseCallback('Requisito 1',get_mouse_position)

cv2.imshow("Requisito 1", imagem)
cv2.waitKey(0)
