import numpy as np
import cv2
import time

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

image_points = []
model_point = []

for n in range (48):
    #O ponto na imagem é mapeado por (xi,yi)
    IMGPoint = imgpoints[n][0][:]
    #O ponto do objeto é mapeado por (Xi, Yi, Zi)
    OBJPoint = objpoints[n][:]

    image_points.append(tuple(IMGPoint))
    model_point.append(tuple(OBJPoint))


image_points = np.asarray(image_points)
model_points = np.asarray(model_point)

camera_matrix = np.array([
    [763.1999, 0.0, 274.03607],
    [0.0, 786.5245, 185.9463],
    [0.0, 0.0, 1.0]
], dtype='double')

distortions = np.array([
    [0.3525], [-0.6317], [-0.01622], [-0.03209], [0.38806]
])

(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, distortions)#, flags=cv2.CV_ITERATIVE)
(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(29.0, 0.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, distortions)

print ("Ponto 1 (imagem) = ({},{})\t Ponto 1 (mundo) = ({},{},{})".format(imgpoints[0][0][0], imgpoints[0][0][1], objpoints[0][0], objpoints[0][1], objpoints[0][2]))
print ("Ponto 2 (imagem) = ({},{})\t Ponto 2 (mundo) = ({},{},{})".format(imgpoints[1][0][0], imgpoints[1][0][1], objpoints[1][0], objpoints[1][1], objpoints[1][2]))

print ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
'''
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

    matrix_coef[n][:] = [OBJPoint[0], OBJPoint[1], OBJPoint[2], 1, OBJPoint[0], OBJPoint[1], OBJPoint[2], 1, -(IMGPoint[0] + IMGPoint[1])*OBJPoint[0], -(IMGPoint[0] + IMGPoint[1])*OBJPoint[1], 
        -(IMGPoint[0] + IMGPoint[1])*OBJPoint[2], -(IMGPoint[0] + IMGPoint[1])]

    #print("Equacao da linha {} = {}".format(n+1, matrix_coef[n][:]))
'''
''' 
# find the eigenvalues and eigenvector of U(transpose).U
e_vals, e_vecs = np.linalg.eig(np.dot(matrix_coef.T, matrix_coef))  
#extract the eigenvector (column) associated with the minimum eigenvalue
p = e_vecs[:, np.argmin(e_vals)] 

xi = (imgpoints[0][0][0] + imgpoints[1][0][0])/2
yi = (imgpoints[0][0][1] + imgpoints[1][0][1])/2
'''

'''
print ("Ponto 1 (imagem) = ({},{})\t Ponto 1 (mundo) = ({},{},{})".format(imgpoints[0][0][0], imgpoints[0][0][1], objpoints[0][0], objpoints[0][1], objpoints[0][2]))
print ("Ponto 2 (imagem) = ({},{})\t Ponto 2 (mundo) = ({},{},{})".format(imgpoints[1][0][0], imgpoints[1][0][1], objpoints[1][0], objpoints[1][1], objpoints[1][2]))

P = np.array(([p[0], p[1], p[2], p[3]], [p[4], p[5], p[6], p[7]], [p[8], p[9], p[10], p[11]]), dtype='float64')
coordIMG = np.array(([xi], [yi], [1]))
coordMundoEsperado = np.array(([14.5], [0.0], [0.0], [1]))
print (np.dot(P,coordMundoEsperado))

print ("Ponto medio = ({},{})".format(xi, yi))
'''