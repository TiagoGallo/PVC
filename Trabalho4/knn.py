import numpy as np
import cv2
import os
import time

def classify(knn, Ponto):
    '''
    Recebe o objeto do OpenCV que aplica o K-NN e um novo ponto(BGR) da imagem,
chama o método para encontrar os vizinhos mais próximos e o label do novo ponto.
Retorna:
    - 0 (ponto do fundo)
    - 1 (ponto de cor de pele)
    '''
    ret, results, neighbours ,dist = knn.findNearest(Ponto, 3)
    
    if results == [1]:
        return 1
    else:
        return 0

def create_imageList_fromDirectory(path):
    '''
    Recebe o path para um diretorio especifico, coloca todos os arquivos com fim .jpg
dele em uma lista e retorna essa lista.
    '''
    filelist=os.listdir(path)
    for fichier in filelist[:]:
        if not(fichier.endswith(".jpg")):
            filelist.remove(fichier)

    aux = []
    for imageName in filelist:
        imagePath = path + imageName
        aux.append(imagePath)

    return aux

#Bool que determina se vai ou nao mostrar na tela as imagens de treino
visualize_train_images = False

#Bool que determina se devemos rodar o algoritmo mais rapido, mas perdendo qualidade
#Caso seja true, nao vai classificar todos os pixels da imagem de treino, vai pular jumpNum pixels para cada classificacao
Run_Faster = False
jumpNum = 100

#Define the train directories paths
GT_Train_Path = './SkinDataset/GT/train/'
Original_Train_path = './SkinDataset/ORI/train/'

#define the test directories paths
GT_Test_Path = './SkinDataset/GT/test/'
Original_Test_path = './SkinDataset/ORI/test/'

#Cria as listas com os paths das imagens de treino e de teste
original_Train_list = create_imageList_fromDirectory(Original_Train_path)
gt_Train_list = create_imageList_fromDirectory(GT_Train_Path)
original_Test_list = create_imageList_fromDirectory(Original_Test_path)
gt_Test_list = create_imageList_fromDirectory(GT_Test_Path)

#print("Lista de treino normal = {}\nLista de treino GT = {}\nLista de teste normal = {}\nLista de teste GT={}".format(original_Train_list, gt_Train_list, original_Test_list, gt_Test_list))

#Define um threshold, pois foi identificada uma falha no ground truth do dataset, algumas imagens nao sao 100% pretas, mas apresentam um ruido
threshold = 27
imgNumber = 0

#Faz um loop por todas as imagens de treino e seus respectivos gt para gerar os dados de treino
for og_image, gt_image in zip(original_Train_list, gt_Train_list):
    #le as imagens
    img_og = cv2.imread(og_image)
    img_gt = cv2.imread(gt_image)

    #Faz um reshape na imagem para virar uma matriz onde cada coluna define uma feature. Nesse caso transforma a matriz no formato (numPixels, 3), onde (:, 1) = 'Blue', (:, 2) = 'Green' e (:, 1) = 'red'
    Z_og = img_og.reshape((-1,3))

    #define os labels
    fundo = 0 #label 0
    pele = 0 # label 1

    #Cria uma lista com o ground truth de cada imagem, sendo que o label 1 significa pele e o label 0 significa fundo
    list_for_gt = []
    for w in range(img_og.shape[0]):
        for h in range(img_og.shape[1]):
            if int(img_gt[w][h][0]) + int(img_gt[w][h][1]) + int(img_gt[w][h][2]) < threshold:
                fundo += 1
                list_for_gt.append(0)
            else:
                pele += 1
                list_for_gt.append(1)
    
    #Transforma essa lista com o ground truth em um numpy array de 1 coluna e NumPixels linhas
    list_for_gt = np.asarray(list_for_gt)
    list_for_gt = np.expand_dims(list_for_gt, axis=1)

    #Caso seja a primeira imagem, criamos o np array que vai ter as informacoes de treino. Para qualquer outra imagem, apenas empilhamos o seu resultado verticalmente no vetor ja existente
    if imgNumber == 0:
        finalData = Z_og
        finalLabels = list_for_gt
    else:
        finalData = np.vstack([finalData, Z_og])
        finalLabels = np.vstack([finalLabels, list_for_gt])

    if visualize_train_images:
        print("Aperte qualquer tecla para passar pra proxima imagem de treino")
        cv2.imshow("original", img_og)
        cv2.imshow("groundTruth", img_gt)
        cv2.waitKey(0)
    
    #incrementa o contador de imagens
    imgNumber += 1

#se tivesse mostrando as imagens, destroi qualquer janela que pode ter ficado aberta
if visualize_train_images:
    cv2.destroyAllWindows()

#Converte os arrays finais de treino para float 32 bits, pois e o formato esperado pelo objeto k-nn do opencv
finalData = np.float32(finalData)
finalLabels = np.float32(finalLabels)

#Cria o objeto de k-nn do opencv e faz o treinamento dele usando os dados de treino e dos labels e indicando que as features estao em colunas
knn = cv2.ml.KNearest_create()
knn.train(finalData, cv2.ml.ROW_SAMPLE, finalLabels)

#le a primeira imagem da lista de imagens de teste para usar elas para classificar
og_img_test = cv2.imread('./SkinDataset/ORI/test/278.jpg')
gt_img_test = cv2.imread('./SkinDataset/GT/test/278.jpg')

#cria uma imagem de classificacao com as mesmas dimensoes que a imagem de teste
classified = og_img_test.copy()

#Calcula algumas variaveis para ajudar a acompanhar o andamento do codigo
Num_a_processar = og_img_test.shape[0] * og_img_test.shape[1]
Num_processado = 0
start_time = time.time()

#Loop em todos os pixels da imagem de test para classificar ela
i = jumpNum
for w in range(og_img_test.shape[0]):
    for h in range(og_img_test.shape[1]):
        #Testa se esta no modo de rodar mais rapido para pular pixels
        if Run_Faster:
            if i % jumpNum != 0:
                Num_processado += 1
                classified[w][h][:] = (0,0,0)
                i += 1
                continue

        #pega a cor do pixel atual e coloca ela no formato esperado pelo knn para poder usar ele como classificador
        cor = og_img_test[w][h][:]
        cor = np.expand_dims(cor, axis=1)
        cor = cor.reshape((-1,3))
        
        #Chama a funcao de classificacao com a cor do pixel atual e pinta o pixel da imagem de classificacao de preto ou branco dependendo do resultado
        label = classify(knn, np.float32(cor))
        if label == 0:
            classified[w][h][:] = (0,0,0)
        else:
            classified[w][h][:] = (255,255,255)

        #Print para acompanhar o processo de teste
        if Num_processado == 0 or Num_processado % int(Num_a_processar / 100) == 0:
            print("Passo demorou: {:.2f}seg.\tJa foram processados {} pontos de um total de {} pontos ===> {:.2f}%".format(time.time() - start_time,Num_processado, Num_a_processar, (Num_processado/Num_a_processar)*100))
            start_time = time.time()

        #incrementa os contadores
        Num_processado +=1
        i += 1

#Pega o nome da imagem e prepara o nome para escrever ela
finalPath = './SkinDataset/Masked/knn/278.jpg'
#imageName = original_Test_list[0].split('/')[-1]
#finalPath += imageName
cv2.imwrite(finalPath, classified)

#Mostra os resultados na tela
print("Aperte qualquer tecla para sair")
cv2.imshow("imagem original", og_img_test)
cv2.imshow("Ground Truth", gt_img_test)
cv2.imshow("Classified", classified)
cv2.waitKey(0)