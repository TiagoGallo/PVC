import numpy as np
import cv2
import time
import os
import xml.etree.cElementTree as ET
from os import listdir
from os.path import isfile, join

class Req3:
    def __init__(self):
        self.tam_quadrado = 29 #milimetros

        self.WebCam = cv2.VideoCapture(0)

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.tam_quadrado, 0.001)

        if not os.path.exists('./images'):
            os.makedirs('./images')

    def get_points(self, num, proximidade):
        if proximidade == 0:
            ImageName = './images/imagemProxima{}.jpg'.format(num+1)
        if proximidade == 1:
            ImageName = './images/imagemMedia{}.jpg'.format(num+1)
        if proximidade == 2:
            ImageName = './images/imagemLonge{}.jpg'.format(num+1)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*8,3), dtype='float32')
        objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.

        foundImageWithChessBoard = False
        
        #Set start time for the detections
        start_time = time.time()
        while not foundImageWithChessBoard:
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
            if ret == True and elapsed > 0.2:
                foundImageWithChessBoard = True
                self.objpoints = objp
                self.objpoints = self.objpoints * self.tam_quadrado

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                self.imgpoints = corners2

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (8,6), corners2,ret)
                cv2.imshow('img',img)
                cv2.imwrite(ImageName, img)

            #Aperte a tecla 'q' para encerrar o programa
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
        
        
        self.image_points = []
        self.model_point = []

        for n in range (48):
            #O ponto na imagem é mapeado por (xi,yi)
            IMGPoint = self.imgpoints[n][0][:]
            #O ponto do objeto é mapeado por (Xi, Yi, Zi)
            OBJPoint = self.objpoints[n][:]

            self.image_points.append(tuple(IMGPoint))
            self.model_point.append(tuple(OBJPoint))


        self.image_points = np.asarray(self.image_points)
        self.model_points = np.asarray(self.model_point)

    def calc_extrinsics(self, n):
        if n == 0:
            print ("Calculando os parametros extrinsicos para uma distancia proxima da camera (dmin)")
        if n == 1:
            print ("Calculando os parametros extrinsicos para uma distancia media da camera (dmed)")
        if n == 2:
            print ("Calculando os parametros extrinsicos para uma distancia longe da camera (dmax)")

        translations = []
        rotacoes = []
        distancias = []
        for i in range(3):
            print("Pegando a imagem numero {}".format(i + 1))
            self.get_points(i, n)
            (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points, self.image_points, self.camera_matrix, self.distortions)
            rot = cv2.Rodrigues(self.rotation_vector)[0]

            dist = np.sqrt((self.translation_vector[0] ** 2) + (self.translation_vector[1] ** 2) + (self.translation_vector[2] ** 2))
            distancias.append(dist)
            translations.append(self.translation_vector)
            rotacoes.append(rot)

        med_dist = (distancias[0] + distancias[1] + distancias[2]) / 3
        desv_dist = np.sqrt((((med_dist - distancias[0]) ** 2) + ((med_dist - distancias[1]) ** 2) + ((med_dist - distancias[2]) ** 2)) / 3)

        print ("A norma do vetor de translacao para  foi {}mm e o desvio {}mm".format(med_dist, desv_dist))
        self.calc_norma_media_desvio(n, translations, rotacoes)

    def calc_norma_media_desvio(self, n, translations, rotacoes):
        translations_media = np.zeros((3,1), dtype='float')

        dists = []
        #calcula o vetor de translacao medio
        for i in range(len(translations)):
            translations_media = np.array(translations_media) + np.array(translations[i])
            dist_trans = np.sqrt((translations[i][0] ** 2) + (translations[i][1] ** 2) + (translations[i][2] ** 2))

        translations_media = np.array(translations_media) / len(translations)


        #Calcula a distancia media e o desvio da distancia media
        dist_media = np.sqrt((translations_media[0] ** 2) + (translations_media[1] ** 2) + (translations_media[2] ** 2))

        desvio = 0
        for i in range(len(translations)):
            desvio += (dist_trans - dist_media) ** 2
        
        desvio = np.sqrt(desvio/len(translations))

        #calcula o vetor de rotacao medio:
        rotacao_media = np.zeros((3,1), dtype='float')

        for i in range(len(rotacoes)):
            rotacao_media = np.array(rotacao_media) + np.array(rotacoes[i])

        rotacao_media = np.array(rotacao_media) / len(rotacoes)
        
        #Verifica para qual distancia os valores foram calculados e armazena isso para botar num xml depois
        if n == 0:
            self.translation_med_perto = translations_media
            self.norma_t_med_perto = dist_media
            self.desvio_t_perto = desvio
            self.rotacao_perto = rotacao_media
            print ("A norma do vetor de translacao para dmin foi {:.2f}mm e o desvio {:.2f}mm".format(dist_media[0], desvio[0]))
        
        elif n == 1:
            self.translation_med_medio = translations_media
            self.norma_t_med_medio = dist_media
            self.desvio_t_medio = desvio
            self.rotacao_medio = rotacao_media
            print ("A norma do vetor de translacao para dmed foi {:.2f}mm e o desvio {:.2f}mm".format(dist_media[0], desvio[0]))
        
        elif n == 2:
            self.translation_med_longe = translations_media
            self.norma_t_med_longe = dist_media
            self.desvio_t_longe = desvio
            self.rotacao_longe = rotacao_media
            print ("A norma do vetor de translacao para dmax foi {:.2f}mm e o desvio {:.2f}mm".format(dist_media[0], desvio[0]))
            
    
    def create_extrinsic_xml(self):
        extrinsic = ET.Element("Calibration_extrinsic_parameters")
        
        min = ET.SubElement(extrinsic, "distancia_minima")
        translation = ET.SubElement(min, "translacao")
        ET.SubElement(translation, "cols").text = '1\n'
        ET.SubElement(translation, "dt").text = 'd\n'
        ET.SubElement(translation, "data_media").text = '{}\t{}\t{}\n'.format(self.translation_med_perto[0][0], self.translation_med_perto[1][0], self.translation_med_perto[2][0])
        ET.SubElement(translation, "data_norma_media").text = str(self.norma_t_med_perto[0]) + '\n'
        ET.SubElement(translation, "data_desvio").text = str(self.desvio_t_perto[0]) + '\n'
        rotacao = ET.SubElement(min, "rotacao")
        ET.SubElement(rotacao, "data_media").text = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(self.rotacao_perto[0][0], self.rotacao_perto[0][1], self.rotacao_perto[0][2],
            self.rotacao_perto[1][0], self.rotacao_perto[1][1], self.rotacao_perto[1][2], self.rotacao_perto[2][0], self.rotacao_perto[2][1], self.rotacao_perto[2][2])

        med = ET.SubElement(extrinsic, "distancia_media")
        translation = ET.SubElement(med, "translacao")
        ET.SubElement(translation, "cols").text = '1\n'
        ET.SubElement(translation, "dt").text = 'd\n'
        ET.SubElement(translation, "data_media").text = '{}\t{}\t{}\n'.format(self.translation_med_medio[0][0], self.translation_med_medio[1][0], self.translation_med_medio[2][0])
        ET.SubElement(translation, "data_norma_media").text = str(self.norma_t_med_medio[0]) + '\n'
        ET.SubElement(translation, "data_desvio").text = str(self.desvio_t_medio[0]) + '\n'
        rotacao = ET.SubElement(med, "rotacao")
        ET.SubElement(rotacao, "data_media").text = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(self.rotacao_medio[0][0], self.rotacao_medio[0][1], self.rotacao_medio[0][2],
            self.rotacao_medio[1][0], self.rotacao_medio[1][1], self.rotacao_medio[1][2], self.rotacao_medio[2][0], self.rotacao_medio[2][1], self.rotacao_medio[2][2])

        max = ET.SubElement(extrinsic, "distancia_longe")
        translation = ET.SubElement(max, "translacao")
        ET.SubElement(translation, "cols").text = '1\n'
        ET.SubElement(translation, "dt").text = 'd\n'
        ET.SubElement(translation, "data_media").text = '{}\t{}\t{}\n'.format(self.translation_med_longe[0][0], self.translation_med_longe[1][0], self.translation_med_longe[2][0])
        ET.SubElement(translation, "data_norma_media").text = str(self.norma_t_med_longe[0]) + '\n'
        ET.SubElement(translation, "data_desvio").text = str(self.desvio_t_longe[0]) + '\n'
        rotacao = ET.SubElement(max, "rotacao")
        ET.SubElement(rotacao, "data_media").text = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(self.rotacao_longe[0][0], self.rotacao_longe[0][1], self.rotacao_longe[0][2],
            self.rotacao_longe[1][0], self.rotacao_longe[1][1], self.rotacao_longe[1][2], self.rotacao_longe[2][0], self.rotacao_longe[2][1], self.rotacao_longe[2][2])

        tree = ET.ElementTree(extrinsic)
        tree.write('./XMLs/extrinsic.xml')
    

    def get_intrinsic_values(self):
        #Faz a leitura do arquivo intrinsics.xml gerado pelo requisito2.py e retira a matriz de valores
        #intrinsicos da camera de lá
        xmlFile = './XMLs/intrisics.xml'
        
        if not (os.path.isfile(xmlFile)):
            raise NameError ('O aquivo intrisics.xml nao existe, por favor execute o programa requisito2.py para que esse arquivo seja gerado')
        
        tree = ET.parse(xmlFile)
        root = tree.getroot()

        for cam_mtx in root.findall('data_media'):
            cam_mtx = cam_mtx.text
            mtx = cam_mtx.split('\t')

            self.camera_matrix = np.array([
                [mtx[0], mtx[1], mtx[2]],
                [mtx[3], mtx[4], mtx[5]],
                [mtx[6], mtx[7], mtx[8]]
            ], dtype='double')

        #Faz a leitura do arquivo distortion.xml gerado pelo requisito2.py e retira a matriz de distorcoes de lá
        xmlFile = './XMLs/distortion.xml'
        if not (os.path.isfile(xmlFile)):
            raise NameError ('O aquivo distortion.xml nao existe, por favor execute o programa requisito2.py para que esse arquivo seja gerado')
        
        tree = ET.parse(xmlFile)
        root = tree.getroot()

        for dist_mtx in root.findall('data_media'):
            dist_mtx = dist_mtx.text
            mtx = dist_mtx.split('\t')

            self.distortions = np.array([
                [mtx[0]], [mtx[1]], [mtx[2]], [mtx[3]], [mtx[4]]
            ], dtype='double')


    def clean_image_directory(self):
        onlyfiles = [f for f in listdir('./images') if isfile(join('./images', f))]

        for file in onlyfiles:
            os.remove('./images/' + file)

    def run(self):
        self.clean_image_directory()
        self.get_intrinsic_values()
        
        for n in range (3):
            self.calc_extrinsics(n)
        
        self.create_extrinsic_xml()

if __name__ == '__main__':
    t = Req3()
    t.run()