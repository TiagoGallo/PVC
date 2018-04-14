import numpy as np
import cv2
import time 
import xml.etree.cElementTree as ET
from os import listdir
from os.path import isfile, join

class Req2:
    def __init__(self):
        self.WebCam = cv2.VideoCapture(1)

        #Contador para o número de imagens onde o xadrez foi detectado
        self.detected_images = 0

        #número de imagens que queremos detectar o tabuleiro de xadrez para
        #calcular os parâmetros intrínsecos da câmera
        self.max_images = 10

    def create_xml(self, num):
        calibration = ET.Element("Calibration intrinsic parameters")
        matrix = ET.SubElement(calibration, "camera_matrix", type_id="opencv-matrix")
        ET.SubElement(matrix, "rows").text = '3\n'
        ET.SubElement(matrix, "cols").text = '3\n'
        ET.SubElement(matrix, "dt").text = 'd\n'
        ET.SubElement(matrix, "data").text = '{}\t{}\t{}\t{}\n{}\t{}\t{}\t{}\t{}\t\n'.format(self.mtx[0,0], self.mtx[0,1], self.mtx[0,2], self.mtx[1,0],
            self.mtx[1,1], self.mtx[1,2], self.mtx[2,0], self.mtx[2,1], self.mtx[2,2])

        distortion = ET.SubElement(calibration, "distortion_coefficients", type_id="opencv-matrix")
        ET.SubElement(distortion, "rows").text = '5\n'
        ET.SubElement(distortion, "cols").text = '1\n'
        ET.SubElement(distortion, "dt").text = 'd\n'
        ET.SubElement(distortion, "data").text = '{}\t{}\t{}\t{}\t{}\t'.format(self.dist[0][0], self.dist[0][1], self.dist[0][2], self.dist[0][3], self.dist[0][4])

        tree = ET.ElementTree(calibration)

        xmlName = './XMLs/calibration{}.xml'.format(num)
        tree.write(xmlName)

    def calibration(self, num):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        #Set start time for the detections
        start_time = time.time()
        while self.detected_images != self.max_images:
            #Ve quant tempo tem desde a ultima deteccao
            elapsed = time.time() - start_time

            grab, img = self.WebCam.read()
            if not grab:
                break

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            cv2.imshow("Webcam", img)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8,6),None)

            # If found, add object points, image points (after refining them)
            if ret == True and elapsed > 0.1:
                self.detected_images += 1
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (8,6), corners2,ret)
                cv2.imshow('img',img)

                #Após detectar o xadrez em uma imagem, da um sleep de 2s para mudar o xadrez de posição
                print("Achei a imagem número {}".format(self.detected_images))
                start_time = time.time()

            #Aperte a tecla 'q' para encerrar o programa
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

        cv2.destroyAllWindows()

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        self.create_xml(num)

    def correct_distortion(self):
        while True:
            grab, img = self.WebCam.read()
            if not grab:
                break

            cv2.imshow("Distorcida", img)
            h,  w = img.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),1,(w,h))

            # undistort
            mapx,mapy = cv2.initUndistortRectifyMap(self.mtx,self.dist,None,newcameramtx,(w,h),5)
            dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

            cv2.imshow('Corrigida',dst)

            #Aperte a tecla 'q' para encerrar o programa
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

        cv2.destroyAllWindows()

def get_xml_id():
    max_id = 0
    onlyfiles = [f for f in listdir('./xMLs') if isfile(join('./xMLs', f))]
    for f in onlyfiles:
        id = f.split('.')[0]
        id = id[-1]
        if int(id) > max_id:
            max_id = int(id)
    return (max_id + 1)

if __name__ == "__main__":
    t = Req2()
    num = get_xml_id()
    t.calibration(num)
    t.correct_distortion()