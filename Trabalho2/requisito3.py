import numpy as np
import cv2
import time
import os
import xml.etree.cElementTree as ET

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
            if ret == True and elapsed > 2:
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
        for i in range(2):
            print("Pegando a imagem numero {}".format(i + 1))
            self.get_points(i, n)
            (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points, self.image_points, self.camera_matrix, self.distortions)

            #(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(29.0, 0.0, 0.0)]), self.rotation_vector, self.translation_vector, self.camera_matrix, self.distortions)

            translations.append(self.translation_vector)
            rotacoes.append(self.rotation_vector)

        self.calc_norma_media_desvio(n, translations, rotacoes)

    def calc_norma_media_desvio(self, n, translations, rotacoes):
        translations_media = np.zeros((3,1), dtype='float')

        #calcula o vetor de translacao medio
        for i in range(len(translations)):
            translations_media = np.array(translations_media) + np.array(translations[i])
        
        translations_media = np.array(translations_media) / len(translations)


        #Calcula a distancia media e o desvio da distancia media
        dist_media = np.sqrt((translations_media[0] ** 2) + (translations_media[1] ** 2) + (translations_media[2] ** 2))
        desvio = 0
        for i in range(len(translations)):
            dist_trans = np.sqrt((translations[i][0] ** 2) + (translations[i][1] ** 2) + (translations[i][2] ** 2))
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
            
    '''
    def create_mean_xml(matrix_media, matrix_desvio, dist_media, dist_desvio):
        extrinsic = ET.Element("Calibration_extrinsic_parameters")
        ET.SubElement(distortion, "rows").text = '5\n'
        ET.SubElement(distortion, "cols").text = '1\n'
        ET.SubElement(distortion, "dt").text = 'd\n'
        ET.SubElement(distortion, "data_media").text = '{}\t{}\t{}\t{}\t{}\n'.format(dist_media[0][0], dist_media[0][1], dist_media[0][2], dist_media[0][3], dist_media[0][4])
        ET.SubElement(distortion, "data_desvio").text = '{}\t{}\t{}\t{}\t{}'.format(dist_desvio[0][0], dist_desvio[0][1], dist_desvio[0][2], dist_desvio[0][3], dist_desvio[0][4])
        tree = ET.ElementTree(distortion)
        tree.write('./XMLs/distortion.xml')
    '''

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


    def run(self):
        self.get_intrinsic_values()
        
        for n in range (3):
            self.calc_extrinsics(n)
        

if __name__ == '__main__':
    t = Req3()
    t.run()