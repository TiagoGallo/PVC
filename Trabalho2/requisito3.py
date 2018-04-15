import numpy as np
import cv2
import time
a = 0
b = 0

def get_mouse_position(event,x,y,flags,param, a,b):
    '''
        Método de callback para clique do mouse, quando ocorrer um clique do mouse na janela
    "Requisito 1" do OpenCV esse método será chamado e vai salvar as coordenadas do mouse na
    tela. A primeira vez que o método for chamado ele vai salvar as coordenadas iniciais do
    objeto, a segunda vez ele salva as coordenadas finais e em todas as outras vezes ele não 
    faz nada.
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        a = x
        b = y

cv2.namedWindow("Webcam")
cv2.namedWindow("img")

cv2.setMouseCallback('img',get_mouse_position)
cv2.setMouseCallback('Webcam',get_mouse_position)

tam_quadrado = 29 #milimetros

WebCam = cv2.VideoCapture(0)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, tam_quadrado, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

while True:
    grab, img = WebCam.read()
    if not grab:
        break

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    cv2.imshow("Webcam", img)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints = objp
        objpoints = objpoints * tam_quadrado

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints = corners2

        image = img.copy()
        # Draw and display the corners
        image = cv2.drawChessboardCorners(image, (8,6), corners2,ret)
        cv2.imshow('img',image)
        break

######################### PREPARE THE LINEAR EQUATIONS #############################
#Cria a matriz dos coeficientes:
matrix_coef = np.zeros((96,12))
matrix_results = np.zeros((96,1))

for n in range (48):
    #O ponto na imagem é mapeado por (xi,yi)
    IMGPoint = imgpoints[n][0][:]
    #O ponto do objeto é mapeado por (Xi, Yi, Zi)
    OBJPoint = objpoints[n][:]

    #A equação linerar para achar p transposto pode ser escrita como:
    # p11*Xi + p12*Yi + p13*Zi + p14 + p21*Xi + p22*Yi + p23*Zi + p24 - (xi+yi)*p31*Xi - (xi+yi)*p32*Yi - (xi+yi)*p33*Zi - (xi+yi)*p34 = 0
    #Logo podemos colocar esses coeficientes em um array para incluirmos eles na matrix dos coeficientes

    matrix_coef[2*n][:] = [OBJPoint[0], OBJPoint[1], OBJPoint[2], 1, 0, 0, 0, 0, -(IMGPoint[0])*OBJPoint[0], -(IMGPoint[0])*OBJPoint[1], 
        -(IMGPoint[0])*OBJPoint[2], -(IMGPoint[0])]
    matrix_coef[2*n + 1][:] = [0, 0, 0, 0, OBJPoint[0], OBJPoint[1], OBJPoint[2], 1, -(IMGPoint[1])*OBJPoint[0], -(IMGPoint[1])*OBJPoint[1], 
        -(IMGPoint[1])*OBJPoint[2], -(IMGPoint[1])]

    #print("Equacao da linha {} = {}".format(n+1, matrix_coef[n][:]))

    
# find the eigenvalues and eigenvector of U(transpose).U
e_vals, e_vecs = np.linalg.eig(np.dot(matrix_coef.T, matrix_coef))  

soma = 100000000000000000
diff_min = 0
a, b =  e_vecs.size
for n in range(b):
    p = e_vecs[:, n]
    diff = 0
    for i in range(48):
        #O ponto na imagem é mapeado por (xi,yi)
        IMGPoint = imgpoints[i][0][:]
        #O ponto do objeto é mapeado por (Xi, Yi, Zi)
        OBJPoint = objpoints[i][:]

        Pi = np.array(([IMGPoint[0]], [IMGPoint[1]], [1]))
        Po = np.array(([OBJPoint[0]], [OBJPoint[1]], [OBJPoint[2]], [1]))

        diff += 
        


#extract the eigenvector (column) associated with the minimum eigenvalue
p = e_vecs[:, np.argmin(e_vals)] 

xi = (imgpoints[0][0][0] + imgpoints[1][0][0])/2
yi = (imgpoints[0][0][1] + imgpoints[1][0][1])/2

print ("Ponto 1 (imagem) = ({},{})\t Ponto 1 (mundo) = ({},{},{})".format(imgpoints[0][0][0], imgpoints[0][0][1], objpoints[0][0], objpoints[0][1], objpoints[0][2]))
print ("Ponto 2 (imagem) = ({},{})\t Ponto 2 (mundo) = ({},{},{})".format(imgpoints[1][0][0], imgpoints[1][0][1], objpoints[1][0], objpoints[1][1], objpoints[1][2]))

P = np.array(([p[0], p[1], p[2], p[3]], [p[4], p[5], p[6], p[7]], [p[8], p[9], p[10], p[11]]), dtype='float64')
coordIMG = np.array(([xi], [yi], [1]))
coordMundoEsperado = np.array(([14.5], [0.0], [0.0], [1]))
print(P)
print (np.dot(P,coordMundoEsperado))

print ("Ponto medio = ({},{})".format(xi, yi))


#aplicando um metodo de solucao numerica para resiolver esse conjunto de equacoes, temos:
#U, s, V = np.linalg.svd(matrix_coef)

# solving Ax=b using the equation above
#c = np.dot(U.T,matrix_results) # c = U^t*b
#w = np.linalg.lstsq(np.diag(s),c) # w = V^t*c
#xSVD = np.dot(V.T,w) # x = V*w

#print (xSVD.T)