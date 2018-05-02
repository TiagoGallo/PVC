import cv2
import numpy as np
import argparse

class mouse_click:
    def __init__(self):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click)
        self.num_cliques = 0

    def click(self, event, x, y, flags, param):
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
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
                print("Os dois pontos ja foram pegos e a distancia calculada, aperte"
                    " 'q' para ir pra proxima imagem")
                pass 

##################################################################################
#                                                                                #
#                 FUNCOES PRINCIPAIS DE CADA REQUISITO                           #
#                                                                                #
##################################################################################
def main_req1(args):
    #Pega o tamanho da janela a partir da entrada do usuario
    W = int(args["Win_size"])
    
    #Carrega os parametros da camera stereo, fornecidos pelo professor
    focal_lenght = 25
    baseline = 120

    #Analisa a entrada do usuário para definir se vamos preparar o mapa de profundidade
    #da imagem do bebê ou da planta
    if args["image"] == "baby":
        imgL = cv2.imread("./imgs-estereo/babyL.png")
        imgR = cv2.imread("./imgs-estereo/babyR.png")
    elif args["image"] == "aloe":
        imgL = cv2.imread("./imgs-estereo/aloeL.png")
        imgR = cv2.imread("./imgs-estereo/aloeR.png")
    else:
        raise NameError ("O parametro -i so pode receber como" 
            "entrada a palavra 'baby' ou  a palavra 'aloe'")

    #Calcula a disparidade entre as imagens
    disp = get_disparity_map(imgL,imgR, W)

    #Cria uma matriz para armazenar os parametros do mundo e calcula eles
    world_coordinates = np.zeros((imgL.shape[1], imgL.shape[0], 3))
    calc_world_coordinates(world_coordinates, focal_lenght, baseline, disp)

    #Normaliza a profundidade
    normalize_depth(world_coordinates)

    depth_image = get_depth_image(world_coordinates, (imgL.shape[0], imgL.shape[1], 1))

    #Aplica um resize nas imagens para facilitar a visualização
    imgL = resize_keep_ar(imgL, cte=3)
    imgR = resize_keep_ar(imgR, cte=3)
    disp = resize_keep_ar(disp, cte=3)
    depth_image = resize_keep_ar(depth_image, cte=3)

    #Mostra todas as imagens na tela
    cv2.imshow("both", np.hstack((imgL, imgR)))
    cv2.imshow("disparity", disp)
    cv2.imshow("profundidade", depth_image)

    print("Aperte qualquer tecla para sair!\n")
    cv2.waitKey(0)

def main_req2(args):
    W = int(args["Win_size"])
    
    intrinsic_MatrixL, intrinsic_MatrixR, points = intrinsic_calibration()

    left_focal_lenght = (intrinsic_MatrixL[1][0][0] + intrinsic_MatrixL[1][1][1]) / 2
    right_focal_lenght = (intrinsic_MatrixR[1][0][0] + intrinsic_MatrixR[1][1][1]) / 2

    focal_lenght = (left_focal_lenght + right_focal_lenght )/ 20

    #print("Lfocal = {}\nRfocal = {}\nfocal = {}".format(left_focal_lenght, right_focal_lenght, focal_lenght))

    #test_undistortion(intrinsic_MatrixL, intrinsic_MatrixR)

    extrinsic_calib_params, baseline = extrinsic_calibration(intrinsic_MatrixL, intrinsic_MatrixR, points)

    Retificated_Images_left, Retificated_Images_Right = retification(extrinsic_calib_params)

    countImages = 0
    for (imgL, imgR) in zip(Retificated_Images_left, Retificated_Images_Right):
        countImages += 1
        disp = get_disparity_map(imgL, imgR, W)
        
        world_coordinates = np.zeros((imgL.shape[1], imgL.shape[0], 3))
        calc_world_coordinates(world_coordinates, focal_lenght, baseline, disp)

        #Normaliza a profundidade
        normalize_depth(world_coordinates)
        
        depth_image = get_depth_image(world_coordinates, (imgL.shape[0], imgL.shape[1], 1))
        cv2.imwrite("./depthImages/req2_pic{}_depth.png".format(countImages), depth_image)
        cv2.imshow("depth", depth_image)
        cv2.waitKey(0)
    
def main_req3(args):
    W = int(args["Win_size"])
    
    intrinsic_MatrixL, intrinsic_MatrixR, points = intrinsic_calibration()

    left_focal_lenght = (intrinsic_MatrixL[1][0][0] + intrinsic_MatrixL[1][1][1]) / 2
    right_focal_lenght = (intrinsic_MatrixR[1][0][0] + intrinsic_MatrixR[1][1][1]) / 2

    focal_lenght = (left_focal_lenght + right_focal_lenght )/ 20

    extrinsic_calib_params, baseline = extrinsic_calibration(intrinsic_MatrixL, intrinsic_MatrixR, points)

    Retificated_Images_left, Retificated_Images_Right = retification(extrinsic_calib_params)

    Mouse = mouse_click()
    countImages = 0
    for (imgL, imgR) in zip(Retificated_Images_left, Retificated_Images_Right):
        countImages += 1
    
        disp = get_disparity_map(imgL, imgR, W)
        
        world_coordinates = np.zeros((imgL.shape[1], imgL.shape[0], 3))
        calc_world_coordinates(world_coordinates, focal_lenght, baseline, disp)

        #Normaliza a profundidade
        normalize_depth(world_coordinates)

        depth_image = get_depth_image(world_coordinates, (imgL.shape[0], imgL.shape[1], 1))
    
        print("Utilizando a imagem de teste numero {}, clique na janela image em 2 pontos"
            " distintos para medir a distancia deles e aperte 'q' para passar"
            " pra proxima imagem")
        while True:
            if Mouse.num_cliques == 2:
                calc_euclidian_distance(world_coordinates, Mouse)
            
            cv2.imshow("depth", depth_image)
            cv2.imshow("image", imgL)
            
            if cv2.waitKey(1) == ord('q'):
                break
        
        Mouse.num_cliques = 0

##################################################################################
#                                                                                #
#                               FUNCOES DE SUPORTE                               #
#                                                                                #
##################################################################################

def resize_keep_ar(imagem, cte = 2.0):
    '''
    Faz um resize na imagem mantendo o aspect ratio dela, ou seja mantendo a mesma
proporcao entre altura e largura da imagem inicial
    Recebe de entrada a imagem a ser modificada e uma constante que determina o 
quanto a imagem vai diminuir
    Retorna a imagem modificada
    '''
    r = cte
    new_w = int(imagem.shape[0]/r)
    new_h = int(imagem.shape[1]/r)

    img = cv2.resize(imagem, (new_w, new_h))

    return img

def get_disparity_map(imgL, imgR, W, max_disparity = 128):
    '''
    Funcao para calcular a disparidade entre duas imagens
    Recebe as duas imagens, o maximo de pixels de disparidade e o tamanho da janela
para calcular o SAD
    Retorna o mapa de disparidade em formato uint8 para permitir visualização
    '''
    print("Calculando a disparidade...\n")
    #Transforma a imagem em preto e branco antes de calcular a disparidade
    imagemL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imagemR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    #Instancia o objeto Stero SGBM do openCV que vai calcular o mapa de disparidade
    stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, W) 
    
    #Calcula a disparidade entre as imagens, utilizando o metodo SAD (sum of absolute
    # diferences) e janela de tamanho WxW definida pelo usuario no inicio do programa
    disparity = stereoProcessor.compute(imagemL,imagemR)

    #Passa um filtro de Speckles para retirar ruídos da imagem 
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity) 

    #Muda a escala do mapa de disparidade, pois o OpenCV retorna ele multiplicado por
    #16 e em um formato não ideal
    disparity_scaled = (disparity / 16.).astype(np.uint8) + abs(disparity.min())

    print("Disparidade calculada...\n")
    return disparity_scaled

def calc_world_coordinates(world_coordinates, focal_lenght, baseline, disp):
    '''
    Calcula as coordenadas do mundo de acordo com o slide 14 da aula 6
    '''
    print("Calculando as coordenadas do mundo...\n")
    for i in range(world_coordinates.shape[1]):
        for j in range(world_coordinates.shape[0]):
            #Pega os pontos x e y nas imagens da esquerda e da direita
            xL = i
            yL = j
            xR = i + disp[i][j]
            yR = j
            #Calcula as coordenadas do mundo
            if (xL- xR) != 0:
                X = (baseline * (xL + xR)) / (2 * (xL- xR))
                Y = (baseline * (yL + yR)) / (2 * (xL- xR))
                Z = (baseline * focal_lenght) / (xL-xR)
            else:
                X = Y = Z = 0
            
            world_coordinates[j][i][0] = X
            world_coordinates[j][i][1] = Y
            world_coordinates[j][i][2] = Z
    print("Coordenadas do mundo calculadas...\n")

def normalize_depth(world_coordinates):
    '''
    Normaliza a coordenada Z dos pontos no mundo, de forma que
os pontos mais distantes tenham valor 1, os mais próximos valor 255
e os que não puderam ser calculados valor 0
    '''
    print("Normalizando a profundidade...")
    Z = world_coordinates[:][:][2]

    Z_min = Z.min()
    Z_max = Z.max()
    Z[:] = Z[:] - Z_min
    Z = Z * 255
    Z[:] = Z[:] / (Z_max - Z_min)

    print("\nZ_max = {}cm\nZ_min = {}cm\n".format(abs(Z_min / 10),abs(Z_max/10)))

    world_coordinates[:][:][2] = Z
    print("Profundidade normalizada")

def get_depth_image(world_coordinates, shape):
    depth_image = np.zeros(shape, dtype='uint8')

    for i in range (shape[0]):
        for j in range (shape[1]):
            if world_coordinates[j][i][2] != 0:
                value = abs(world_coordinates[j][i][2] - 256)
                depth_image[i][j] = value

    return depth_image

def intrinsic_calibration():
    print ("Comecando calibracao dos instrinsecos...\n")
    termination_criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepara um conjubt de pontos do mundo para o padrão do xadrez
    patternX = 8
    patternY = 6 
    square_size_in_mm = 29 

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((patternX*patternY,3), np.float32)
    objp[:,:2] = np.mgrid[0:patternX,0:patternY].T.reshape(-1,2)
    objp = objp * square_size_in_mm 

    # Arrays para armazenar os pontos do objeto e da imagem
    objpoints = [] # 3d point in real world space
    imgpointsR = [] # 2d points in image plane.
    imgpointsL = [] # 2d points in image plane.

    #Contador de imagens
    countImages = 0

    while countImages != 20:
        countImages += 1
        imgL = cv2.imread("./calibrationImages/calibrateLeft{}.png".format(countImages))
        imgR = cv2.imread("./calibrationImages/calibrateRight{}.png".format(countImages))

        # Converte as imagens para grayscae
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY) 
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

        #Encontra o padrao do xadrez nas images
        retR, cornersL = cv2.findChessboardCorners(grayL, (patternX,patternY),None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE) 
        retL, cornersR = cv2.findChessboardCorners(grayR, (patternX,patternY),None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)

        # Se encontrar os pontos, refina eles e mostra a imagem com o padrao detectado

        if ((retR == True) and (retL == True)):
            # adiciona os pontos do objeto a lista
            objpoints.append(objp) 

            # refina a localizacao dos pontos dos cantos do xadrez
            corners_sp_L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),termination_criteria_subpix) 
            imgpointsL.append(corners_sp_L) 
            corners_sp_R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),termination_criteria_subpix) 
            imgpointsR.append(corners_sp_R) 

            # Desenha o padrao na imagem
            drawboardL = cv2.drawChessboardCorners(imgL, (patternX,patternY), corners_sp_L,retL)
            drawboardR = cv2.drawChessboardCorners(imgR, (patternX,patternY), corners_sp_R,retR)

            cv2.imshow("Imagem_Calib_Left",drawboardL) 
            cv2.imshow("Imagem_Calib_Right",drawboardR)

        #Espera 0.5s para mostrar a proxima imagem
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    ret, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1],None,None) 
    ret, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1],None,None) 

    #print("matrix int left = ", mtxL)
    #print("matrix int right = ", mtxR)

    print ("Calibracao dos intrinsecos terminada...\n")

    return((ret, mtxL, distL, rvecsL, tvecsL), (ret, mtxR, distR, rvecsR, tvecsR), (objpoints, imgpointsL, imgpointsR))

def test_undistortion(intrinsic_MatrixL, intrinsic_MatrixR):
    ret, mtxL, distL, rvecsL, tvecsL = intrinsic_MatrixL
    ret, mtxR, distR, rvecsR, tvecsR = intrinsic_MatrixR
    
    camL = cv2.VideoCapture(1)
    camR = cv2.VideoCapture(0)

    print("Testando o undistortion\nAperte 'q' para sair")

    while True:
        camL.grab() 
        camR.grab() 

        ret, frameL = camL.retrieve() 
        ret, frameR = camR.retrieve() 

        undistortedL = cv2.undistort(frameL, mtxL, distL, None, None)
        undistortedR = cv2.undistort(frameR, mtxR, distR, None, None)

        cv2.imshow("camL",undistortedL) 
        cv2.imshow("camR",undistortedR) 

        key = cv2.waitKey(40) & 0xFF  

        if (key == ord('q')):
            break
    cv2.destroyAllWindows()

def calc_reprojection_error():
    tot_errorL = 0
    '''
    for i in range(len(objpoints)):
        imgpointsL2, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
        errorL = cv2.norm(imgpointsL[i],imgpointsL2, cv2.NORM_L2)/len(imgpointsL2)
        tot_errorL += errorL

    print("LEFT: Re-projection error: ", tot_errorL/len(objpoints))

    tot_errorR = 0
    for i in range(len(objpoints)):
        imgpointsR2, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
        errorR = cv2.norm(imgpointsR[i],imgpointsR2, cv2.NORM_L2)/len(imgpointsR2)
        tot_errorR += errorR
    
    print("RIGHT: Re-projection error: ", tot_errorR/len(objpoints))
    '''

def extrinsic_calibration(intrinsic_MatrixL, intrinsic_MatrixR, points):
    ret, mtxL, distL, rvecsL, tvecsL = intrinsic_MatrixL
    ret, mtxR, distR, rvecsR, tvecsR = intrinsic_MatrixR
    objpoints, imgpointsL, imgpointsR = points

    #le a primeira imagem de calibracao apenas para pegar o shape dela 
    grayL = cv2.imread("./calibrationImages/calibrateLeft1.png",0)

    print("Comecando calibracao dos parametros extrinsicos...")
    termination_criteria_extrinsics = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    (rms_stereo, camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R, T, E, F) = \
    cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR,  grayL.shape[::-1], criteria=termination_criteria_extrinsics, flags=0) 

    print("Calibracao dos extrinsicos terminada...")

    #print("STEREO: RMS left to  right re-projection error: ", rms_stereo)

    #Calcula baseline a partir da matriz de translacao entre as cameras
    #print("T = ", T)
    baseline = ((T[0] ** 2) + (T[1] ** 2) + (T[2] ** 2)) ** (1/2)
    #print("baseline = {}mm".format(baseline[0])) 
    baseline = baseline[0]

    return ((rms_stereo, camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R, T, E, F), baseline)

def retification(extrinsic_calib_params):
    rms_stereo, camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R, T, E, F = extrinsic_calib_params
    
    #le a primeira imagem de calibracao apenas para pegar o shape dela 
    grayL = cv2.imread("./calibrationImages/calibrateLeft1.png",0)
    
    RL, RR, PL, PR, _, _, _ = cv2.stereoRectify(camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r,  grayL.shape[::-1], R, T, alpha=-1) 

    # compute the pixel mappings to the rectified versions of the images

    mapL1, mapL2 = cv2.initUndistortRectifyMap(camera_matrix_l, dist_coeffs_l, RL, PL, grayL.shape[::-1], cv2.CV_32FC1) 
    mapR1, mapR2 = cv2.initUndistortRectifyMap(camera_matrix_r, dist_coeffs_r, RR, PR, grayL.shape[::-1], cv2.CV_32FC1) 

    #Cria uma lista de imagens retificadas
    Retificated_Images_left = []
    Retificated_Images_Right = []

    countImages = 0
    while countImages != 5:
        countImages += 1
        imgL = cv2.imread("./testeImages/testeLeft{}.png".format(countImages))
        imgR = cv2.imread("./testeImages/testeRight{}.png".format(countImages))

        # Retira a distorcao e retifica a imagem baseado no mapeamento

        undistorted_rectifiedL = cv2.remap(imgL, mapL1, mapL2, cv2.INTER_LINEAR) 
        undistorted_rectifiedR = cv2.remap(imgR, mapR1, mapR2, cv2.INTER_LINEAR)

        #mostra as imagens
        cv2.imshow("Left_Retificated", undistorted_rectifiedL)
        cv2.imshow("Right_Retificated", undistorted_rectifiedR)

        Retificated_Images_left.append(undistorted_rectifiedL)
        Retificated_Images_Right.append(undistorted_rectifiedR)

        #Espera 2s para mostrar o proximo conjundo de imagens
        cv2.waitKey(2000)

    cv2.destroyAllWindows()

    return (Retificated_Images_left, Retificated_Images_Right)

def calc_euclidian_distance(world_coordinates, Mouse):
    xi = Mouse.xi
    yi = Mouse.yi
    xf = Mouse.xf
    yf = Mouse.yf

    print ("coordenadas do mundo do primeiro ponto = ", world_coordinates[xi][yi][:])
    print ("coordenadas do mundo do segundo ponto = ", world_coordinates[xf][yf][:])

    distX = (world_coordinates[xi][yi][0] - world_coordinates[xf][yf][0]) ** 2
    distY = (world_coordinates[xi][yi][1] - world_coordinates[xf][yf][1]) ** 2
    distZ = (world_coordinates[xi][yi][2] - world_coordinates[xf][yf][2]) ** 2
    tam = (distX + distY + distZ) ** (1/2)

    print("O tamanho do objeto eh {}cm".format(tam/10))
    Mouse.num_cliques += 1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("-r", "--requisito", default=1,
        help="Qual o requisito a ser analisado?")
    ap.add_argument("-w", "--Win_size", default= 13,
        help="Qual o tamanho da janela para o SAD no " 
        "calculo da disparidade?")
    ap.add_argument("-i", "--image", default="aloe",
        help="Qual conjunto de imagens a ser usado para o req 1?")

    args = vars(ap.parse_args())

    #Verifica se o valor inserido para o Win_size eh permitido
    if int(args["Win_size"]) % 2 == 0:
        raise NameError("O parametro -w deve receber um numero impar")

    #Verifica qual requisito a ser executado:
    if int(args["requisito"]) == 1:
        main_req1(args)
    elif int(args["requisito"]) == 2:
        main_req2(args)
    elif int(args["requisito"]) == 3:
        main_req3(args)
    else:
        raise NameError ("O parametro -r so pode receber um numero de 1 a 3")