import numpy as np
import cv2
import time 
import xml.etree.cElementTree as ET
from os import listdir
from os.path import isfile, join

class Req2:
    def __init__(self):
        self.WebCam = cv2.VideoCapture(0)

        # contador para o numero de imagens onde o xadrez foi detectado
        self.detected_images = 0

        #numero de imagens que queremos detectar o tabuleiro de xadrez para
        #calcular os parametros intrinsecos da camera
        self.max_images = 10

        cv2.namedWindow("raw")
        cv2.namedWindow("undistorted")

        cv2.setMouseCallback('raw',self.get_mouse_position)

        #inicializa coordenadas iniciais e finais e o numero de cliques que ja ocorreram
        self.xi = None
        self.yi = None
        self.xf = None
        self.yf = None
        self.num_cliques = 0

    def get_mouse_position(self, event,x,y,flags,param):
        '''
            Método de callback para clique do mouse, quando ocorrer um clique do mouse na janela
        "Requisito 1" do OpenCV esse método será chamado e vai salvar as coordenadas do mouse na
        tela. A primeira vez que o método for chamado ele vai salvar as coordenadas iniciais do
        objeto, a segunda vez ele salva as coordenadas finais e em todas as outras vezes ele não 
        faz nada.
        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.num_cliques == 0:
                self.xi = x
                self.yi = y
                self.num_cliques += 1
                print("Primeiro clique na coordenada ({},{})".format(x,y))

            elif self.num_cliques == 1:
                self.xf = x
                self.yf = y
                self.num_cliques += 1
                print("Segundo clique na coordenada ({},{})".format(x,y))

            else:
                print("Os dois pontos ja foram pegos e a distancia calculada, execute o programa "
                    "novamente para poder escolher dois novos pontos")
                pass


    def create_xml(self, num):
        calibration = ET.Element("Calibration_intrinsic_parameters")
        matrix = ET.SubElement(calibration, "camera_matrix", type_id="opencv-matrix")
        ET.SubElement(matrix, "rows").text = '3\n'
        ET.SubElement(matrix, "cols").text = '3\n'
        ET.SubElement(matrix, "dt").text = 'd\n'
        ET.SubElement(matrix, "data").text = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(self.mtx[0,0], self.mtx[0,1], self.mtx[0,2], self.mtx[1,0],
            self.mtx[1,1], self.mtx[1,2], self.mtx[2,0], self.mtx[2,1], self.mtx[2,2])

        distortion = ET.SubElement(calibration, "distortion_coefficients", type_id="opencv-matrix")
        ET.SubElement(distortion, "rows").text = '5\n'
        ET.SubElement(distortion, "cols").text = '1\n'
        ET.SubElement(distortion, "dt").text = 'd\n'
        ET.SubElement(distortion, "data").text = '{}\t{}\t{}\t{}\t{}'.format(self.dist[0][0], self.dist[0][1], self.dist[0][2], self.dist[0][3], self.dist[0][4])

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

                #Apos detectar o xadrez em uma imagem, da um sleep de 2s para mudar o xadrez de posicao
                start_time = time.time()

            #Aperte a tecla 'q' para encerrar o programa
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

        cv2.destroyWindow("img")
        cv2.destroyWindow("Webcam")

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        self.create_xml(num)

    def correct_distortion(self):
        grab, img = self.WebCam.read()
        h,  w = img.shape[:2]
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),1,(w,h))
        
        # undistort
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.mtx,self.dist,None,newcameramtx,(w,h),5)
        
        while True:
            grab, img = self.WebCam.read()
            if not grab:
                break

            #Se já tiver pego as duas coordenadas desejadas, desenha a linha na imagem
            if self.num_cliques > 1:
                cv2.line(img,(self.xi,self.yi),(self.xf,self.yf),(255,0,0),5)

            cv2.imshow("raw", img)
            
            #remapeamento
            dst = cv2.remap(img,self.mapx, self.mapy,cv2.INTER_LINEAR)

            cv2.imshow('undistorted', dst)

            if self.num_cliques == 2:
                self.euclidian_distance(dst)
                self.num_cliques += 1

            #Aperte a tecla 'q' para encerrar o programa
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

        cv2.destroyAllWindows()

    def euclidian_distance(self, dst):
        '''
            Método para calcular a distância eucliadiana bidimensional entre dois pontos da imagem,
        será chamado quando clicarmos no segundo ponto da tela e vai imprimir o resultado no console.
        '''
        dx = (self.xi - self.xf) ** 2
        dy = (self.yi - self.yf) ** 2

        dist = np.sqrt(dx + dy)

        '''
            Para a imagem sem distorçao temos que o ponto (xi, yi) foi mapeado em self.mapx(xi, yi)
        '''

        for i in range(self.mapx.shape[0]):
            for j in range(self.mapx.shape[1]):
                if int(self.mapx[i][j]) == self.xi and int(self.mapy[i][j]) == self.yi:
                    xi_calib = j
                    yi_calib = i
                    print ("Ponto inicial na imagem sem distorcao = ({},{})".format(xi_calib, yi_calib))
        
        for i in range(self.mapx.shape[0]):
            for j in range(self.mapx.shape[1]):
                if int(self.mapx[i][j]) == self.xf and int(self.mapy[i][j]) == self.yf:
                    xf_calib = j
                    yf_calib = i
                    print ("Ponto final na imagem sem distorcao = ({},{})".format(xf_calib, yf_calib))

        dx_calib = (xi_calib - xf_calib) ** 2
        dy_calib = (yi_calib - yf_calib) ** 2

        dist_calib = np.sqrt(dx_calib + dy_calib)

        print("A distancia euclidiana dos pontos foi na imagem distorcida foi {}".format(dist))
        print("A distancia euclidiana dos pontos foi na imagem sem distorcao foi {}".format(dist_calib))

def get_xml_id():
    max_id = 0
    onlyfiles = [f for f in listdir('./XMLs') if isfile(join('./XMLs', f))]
    for f in onlyfiles:
        if f[:11] == 'calibration':
            id = f.split('.')[0]
            id = id[11:]
            if int(id) > max_id:
                max_id = int(id)
    return (max_id + 1)

def xml_average():
    matrizes = []
    distorcoes = []
    onlyfiles = [f for f in listdir('./XMLs') if isfile(join('./XMLs', f))]
    for f in onlyfiles:
        if f[:11] == "calibration":
            mtx, dist = get_xml_values(f)
            matrizes.append(mtx)
            distorcoes.append(dist)

    matrix_media, matrix_desvio, dist_media, dist_desvio = calc_media_and_desvio(matrizes, distorcoes)
    create_mean_xml(matrix_media, matrix_desvio, dist_media, dist_desvio)

def get_xml_values(f):
    xmlFile = './XMLs/' + f
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    
    for cam_mtx in root.findall('camera_matrix'):
        mtx = cam_mtx.find('data').text
        mtx = mtx.split('\t')
        matrix = np.zeros((3,3), dtype=float)
        for n in range(9):
            linha = n // 3
            coluna = n % 3
            matrix[linha][coluna] = mtx[n]

    for cam_dist in root.findall('distortion_coefficients'):
        dist = cam_dist.find('data').text
        dist = dist.split('\t')
        distorcao = np.zeros((1,5), dtype=float)
        for n in range(5):
            distorcao[0][n] = dist[n]

    return matrix, distorcao

def create_mean_xml(matrix_media, matrix_desvio, dist_media, dist_desvio):
    distortion = ET.Element("Calibration_distortion_coefficients")
    ET.SubElement(distortion, "rows").text = '5\n'
    ET.SubElement(distortion, "cols").text = '1\n'
    ET.SubElement(distortion, "dt").text = 'd\n'
    ET.SubElement(distortion, "data_media").text = '{}\t{}\t{}\t{}\t{}\n'.format(dist_media[0][0], dist_media[0][1], dist_media[0][2], dist_media[0][3], dist_media[0][4])
    ET.SubElement(distortion, "data_desvio").text = '{}\t{}\t{}\t{}\t{}'.format(dist_desvio[0][0], dist_desvio[0][1], dist_desvio[0][2], dist_desvio[0][3], dist_desvio[0][4])
    tree = ET.ElementTree(distortion)
    tree.write('./XMLs/distortion.xml')


    intrinsic = ET.Element("Calibration_intrinsic_parameters")
    ET.SubElement(intrinsic, "rows").text = '3\n'
    ET.SubElement(intrinsic, "cols").text = '3\n'
    ET.SubElement(intrinsic, "dt").text = 'd\n'
    ET.SubElement(intrinsic, "data_media").text = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(matrix_media[0,0], matrix_media[0,1], matrix_media[0,2], matrix_media[1,0],
        matrix_media[1,1], matrix_media[1,2], matrix_media[2,0], matrix_media[2,1], matrix_media[2,2])
    ET.SubElement(intrinsic, "data_desvio").text = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(matrix_desvio[0,0], matrix_desvio[0,1], matrix_desvio[0,2], matrix_desvio[1,0],
        matrix_desvio[1,1], matrix_desvio[1,2], matrix_desvio[2,0], matrix_desvio[2,1], matrix_desvio[2,2])
    tree = ET.ElementTree(intrinsic)
    tree.write('./XMLs/intrisics.xml')

def calc_media_and_desvio(matrizes, distorcoes):
    matrix_media = np.zeros((3,3), dtype=float)
    dist_media = np.zeros(5, dtype=float)
    for n in range(len(matrizes)):
        matrix_media = np.array(matrix_media) + np.array(matrizes[n])
        dist_media = np.array(dist_media) + np.array(distorcoes[n])

    matrix_media = np.array(matrix_media)/len(matrizes)
    dist_media = np.array(dist_media)/len(distorcoes)

    matrix_desvio = np.zeros((3,3), dtype=float)
    dist_desvio = np.zeros(5, dtype=float)
    for n in range(len(matrizes)):
        matrix_desvio = np.power(np.array(matrizes[n]) - np.array(matrix_media), 2)
        dist_desvio = np.power(np.array(distorcoes[n]) - np.array(dist_media), 2)

    matrix_desvio = np.array(matrix_desvio)/len(matrizes)
    matrix_desvio = np.sqrt(matrix_desvio)
    dist_desvio = np.array(dist_desvio)/len(distorcoes)
    dist_desvio = np.sqrt(dist_desvio)

    return matrix_media, matrix_desvio, dist_media, dist_desvio 

if __name__ == "__main__":
    t = Req2()
    num = get_xml_id()
    t.calibration(num)
    t.correct_distortion()

    xml_average()