import numpy as np
import cv2
import time

class Req3:
    def __init__(self):
        self.tam_quadrado = 29 #milimetros

        self.WebCam = cv2.VideoCapture(0)

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.tam_quadrado, 0.001)

    def get_points(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.

        while True:
            grab, img = self.WebCam.read()
            if not grab:
                break

            cv2.imshow("Webcam", img)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8,6),None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints = objp
                self.objpoints = self.objpoints * self.tam_quadrado

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), self.criteria)
                self.imgpoints = corners2

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (8,6), corners2,ret)
                cv2.imshow('img',img)
                time.sleep(2.0)
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
        for i in range(3):
            print("Pegando a imagem numero {}".format(i + 1))
            self.get_points()
            (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points, self.image_points, self.camera_matrix, self.distortions)
            dist = np.sqrt((self.translation_vector[0] ** 2) + (self.translation_vector[1] ** 2) + (self.translation_vector[2] ** 2))
            translations.append(dist)
        
        med_dist = (translations[0] + translations[1] + translations[2]) / 3
        desv_dist = np.sqrt((((med_dist - translations[0]) ** 2) + ((med_dist - translations[1]) ** 2) + ((med_dist - translations[2]) ** 2)) / 3)

        print("A media da norma do vetor de translacao t eh = {}\te o desvio eh = {}".format(med_dist, desv_dist))

        
        
        #(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(29.0, 0.0, 0.0)]), self.rotation_vector, self.translation_vector, self.camera_matrix, self.distortions)

    def create_mean_xml(matrix_media, matrix_desvio, dist_media, dist_desvio):
        extrinsic = ET.Element("Calibration_extrinsic_parameters")
        ET.SubElement(distortion, "rows").text = '5\n'
        ET.SubElement(distortion, "cols").text = '1\n'
        ET.SubElement(distortion, "dt").text = 'd\n'
        ET.SubElement(distortion, "data_media").text = '{}\t{}\t{}\t{}\t{}\n'.format(dist_media[0][0], dist_media[0][1], dist_media[0][2], dist_media[0][3], dist_media[0][4])
        ET.SubElement(distortion, "data_desvio").text = '{}\t{}\t{}\t{}\t{}'.format(dist_desvio[0][0], dist_desvio[0][1], dist_desvio[0][2], dist_desvio[0][3], dist_desvio[0][4])
        tree = ET.ElementTree(distortion)
        tree.write('./XMLs/distortion.xml')

    def get_intrinsic_values(self):
        self.camera_matrix = np.array([
            [763.1999, 0.0, 274.03607],
            [0.0, 786.5245, 185.9463],
            [0.0, 0.0, 1.0]
        ], dtype='double')

        self.distortions = np.array([
            [0.3525], [-0.6317], [-0.01622], [-0.03209], [0.38806]
        ])

    def run(self):
        self.get_intrinsic_values()
        
        for n in range (3):
            self.calc_extrinsics(n)
        

if __name__ == '__main__':
    Req3().run()