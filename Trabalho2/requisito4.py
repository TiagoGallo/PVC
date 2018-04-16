import os
import cv2
import numpy as np
import argparse
from os.path import isfile
import xml.etree.cElementTree as ET

class Req4:
    def __init__(self, args):
        cv2.namedWindow("raw")
        cv2.setMouseCallback('raw',self.get_mouse_position)
        self.num_cliques = 0

        if args["position"] == "perto":
            self.imageName = './images/imagemProxima{}.jpg'.format(str(args["number"]))
        elif args["position"] == "medio":
            self.imageName = './images/imagemMedia{}.jpg'.format(str(args["number"]))
        elif args["position"] == "longe":
            self.imageName = './images/imagemLonge{}.jpg'.format(str(args["number"]))
        else:
            raise NameError ("O argumento --position so pode receber como parametro 'perto', 'medio' ou 'longe',"
                " {} nao e um parametro valido".format(args["position"]))
        
        if not (os.path.isfile(self.imageName)):
            raise NameError ("A imagem {} desejada nao existe, provavelmente o erro esta no parametro --number passado, "
                "por favor corrija esse valor ou deixe ele como padrao (1)".format(self.imageName))

        self.get_intrinsic_values()
        self.get_extrinsic_values(args)
        self.constructProjectionMatrix()
        self.map((400,150))

        #print (self.imageName)
        #print("Intrinsic values:\nCamera Matrix = {}\nDistortions = {}".format(self.camera_matrix, self.distortions))
        #print("Extrinsic Values:\ntranslation Matrix = {}\nRotation Matrix = {}".format(self.translation_vector, self.rotation_matrix))

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
                #self.calc_euclidian_distance()

            else:
                print("Os dois pontos ja foram pegos e a distancia calculada, execute o programa "
                    "novamente para poder escolher dois novos pontos")
                pass

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

    def get_extrinsic_values(self, args):
        if args["position"] == "perto":
            posName = "distancia_minima"
        elif args["position"] == "medio":
            posName = "distancia_media"
        elif args["position"] == "longe":
            posName = "distancia_longe"

        #Faz a leitura do arquivo extrinsic.xml gerado pelo requisito2.py e retira a matriz de valores
        #intrinsicos da camera de lá
        xmlFile = './XMLs/extrinsic.xml'
        
        if not (os.path.isfile(xmlFile)):
            raise NameError ('O aquivo extrinsic.xml nao existe, por favor execute o programa requisito3.py para que esse arquivo seja gerado')
        
        tree = ET.parse(xmlFile)
        root = tree.getroot()

        for params in root.findall(posName):
            for trans in params.findall("translacao"):
                t_med = trans.find("data_media").text
                t_med = t_med.split('\t')

                self.translation_vector = np.array([
                    [t_med[0]], [t_med[1]], [t_med[2]]
                ], dtype='double')

            for rot in params.findall("rotacao"):
                rot_med = rot.find("data_media").text
                rot_med = rot_med.split('\t')

                #eliminando a 3 coluna, pois Z = 0 sempre
                self.rotation_matrix = np.array([
                    [rot_med[0], rot_med[1]],
                    [rot_med[3], rot_med[4]],
                    [rot_med[6], rot_med[7]]
                ], dtype='float')

    def constructProjectionMatrix(self):
        conc = np.hstack((self.rotation_matrix, self.translation_vector))
        self.P = np.dot(self.camera_matrix, conc)
        self.Pinv = np.linalg.inv(self.P)

    def map(self, point2d):
        #(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(29.0, 0.0, 0.0)]), self.rotation_matrix, self.translation_vector, self.camera_matrix, self.distortions)
        imagePoint = np.array([
            [point2d[0]], [point2d[1]], [1]
        ])

        point3d = np.dot(self.Pinv, imagePoint)
        
        return point3d

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
        assignInit = False
        for i in range(self.mapx.shape[0]):
            for j in range(self.mapx.shape[1]):
                if int(self.mapx[i][j]) == self.xi and int(self.mapy[i][j]) == self.yi:
                    xi_calib = j
                    yi_calib = i
                    assignInit = True
                    print ("Ponto inicial na imagem sem distorcao = ({},{})".format(xi_calib, yi_calib))
        
        assignFinal = False
        for i in range(self.mapx.shape[0]):
            for j in range(self.mapx.shape[1]):
                if int(self.mapx[i][j]) == self.xf and int(self.mapy[i][j]) == self.yf:
                    xf_calib = j
                    yf_calib = i
                    assignFinal = True
                    print ("Ponto final na imagem sem distorcao = ({},{})".format(xf_calib, yf_calib))

        if not(assignInit and assignFinal):
            raise AssertionError ("Houve um problema com o mapeamento sem distorcao para com distorcao, por favor tente novamente")

        dx_calib = (xi_calib - xf_calib) ** 2
        dy_calib = (yi_calib - yf_calib) ** 2

        dist_calib = np.sqrt(dx_calib + dy_calib)

        print("A distancia euclidiana dos pontos foi na imagem distorcida foi {}".format(dist))
        print("A distancia euclidiana dos pontos foi na imagem sem distorcao foi {}".format(dist_calib))

        pontoDistorcidoInit = (self.xi, self.yi)
        pontoDistorcidoFim = (self.xf, self.yf)
        pontoArrumadoInit = (xi_calib, yi_calib)
        pontoArrumadoFim = (xf_calib, yf_calib) 

        return pontoDistorcidoInit, pontoDistorcidoFim, pontoArrumadoInit, pontoArrumadoFim

    def run(self):
        image = cv2.imread(self.imageName)
        h,  w = image.shape[:2]
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix,self.distortions,(w,h),1,(w,h))
        
        # undistort
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.camera_matrix,self.distortions,None,newcameramtx,(w,h),5)
        cv2.imshow("raw", image)

        #remapeamento
        dst = cv2.remap(image,self.mapx, self.mapy,cv2.INTER_LINEAR)

        while True:           
            cv2.imshow("raw", image)
            cv2.imshow("undisorted", dst)

            if self.num_cliques == 2:
                pdi, pdf, pai, paf = self.euclidian_distance(dst)
                ponto1_3d = self.map(pdi)
                ponto2_3d = self.map(pdf)
                ponto1_3d = (ponto1_3d) / ponto1_3d[2]
                ponto2_3d = (ponto2_3d) / ponto2_3d[2]
                tam = np.sqrt(((ponto1_3d[0] - ponto2_3d[0]) ** 2) + ((ponto1_3d[1] - ponto2_3d[1]) ** 2))
                cv2.line(image,pdi,pdf,(255,0,0),5)
                cv2.putText(image,'{:.3f}m'.format(tam[0]/1000),(int((pdi[0]+pdf[0])/2), int((pdi[1]+pdf[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2,cv2.LINE_AA)
                print("O tamanho do objeto na imagem distorcida eh {}m".format(tam[0]/1000))

                ponto1_3d = self.map(pai)
                ponto2_3d = self.map(paf)
                ponto1_3d = (ponto1_3d) / ponto1_3d[2]
                ponto2_3d = (ponto2_3d) / ponto2_3d[2]
                tam = np.sqrt(((ponto1_3d[0] - ponto2_3d[0]) ** 2) + ((ponto1_3d[1] - ponto2_3d[1]) ** 2))
                cv2.line(dst,pai,paf,(255,0,0),5)
                cv2.putText(dst,'{:.3f}m'.format(tam[0]/1000),(int((pai[0]+paf[0])/2), int((pai[1]+paf[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2,cv2.LINE_AA)
                print("O tamanho do objeto na imagem nao distorcida eh {}m".format(tam[0]/1000))
                self.num_cliques += 1

            #Aperte a tecla 'q' para encerrar o programa
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            
            #cv2.imshow("raw", self.image)
            #cv2.imshow("undistorted", self.imagemCalibrada)

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--position", default="perto",
        help="Qual distancia da imagem deseja ser analisada")
    ap.add_argument("-n", "--number", default= 1,
        help="Qual o numero da imagem deseja ser analisada")
    args = vars(ap.parse_args())

    t = Req4(args)
    t.run()
