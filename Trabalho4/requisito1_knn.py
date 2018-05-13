import numpy as np
import cv2
import os
import time

def classify(knn, Ponto):
    ret, results, neighbours ,dist = knn.findNearest(Ponto, 3)
    
    if results == [1]:
        return 1
    else:
        return 0

gt_path = './SkinDataset/GT/'
original_path = './SkinDataset/ORI/'

filelist=os.listdir(gt_path)
for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
    if not(fichier.endswith(".jpg")):
        filelist.remove(fichier)

gt_list = []
original_list = []
for image in filelist:
    gt_image = gt_path + image
    original_image = original_path + image

    gt_list.append(gt_image)
    original_list.append(original_image)

threshold = 27
imgNumber = 0
for og_image, gt_image in zip(original_list[:-2], gt_list[:-2]):
    img_og = cv2.imread(og_image)
    img_gt = cv2.imread(gt_image)

    #print("imagem original shape = ", img_og.shape)
    Z_og = img_og.reshape((-1,3))

    fundo = 0 #label 0
    pele = 0 # label 1
    list_for_gt = []
    i = 0
    for w in range(img_og.shape[0]):
        for h in range(img_og.shape[1]):
            if int(img_gt[w][h][0]) + int(img_gt[w][h][1]) + int(img_gt[w][h][2]) < threshold:
                fundo += 1
                list_for_gt.append(0)
            else:
                pele += 1
                list_for_gt.append(1)
            i += 1

    #print ("Z_og shape = ", Z_og.shape)
    
    list_for_gt = np.asarray(list_for_gt)
    list_for_gt = np.expand_dims(list_for_gt, axis=1)

    #print("List of ground truths = {} and shape = {}".format(list_for_gt, list_for_gt.shape))

    #Z = np.hstack([Z_og, list_for_gt])
    #print("final data shape = ", Z.shape)

    if imgNumber == 0:
        finalData = Z_og
        finalLabels = list_for_gt
    else:
        finalData = np.vstack([finalData, Z_og])
        finalLabels = np.vstack([finalLabels, list_for_gt])

    #cv2.imshow("original", img_og)
    #cv2.imshow("groundTruth", img_gt)
    #print("Tiveram {} pixels de fundo e {} pixels de cor de pele".format(fundo, pele))
    #cv2.waitKey(0)
    imgNumber += 1


#print ("FinalData shape = ", finalData.shape)
finalData = np.float32(finalData)
finalLabels = np.float32(finalLabels)



knn = cv2.ml.KNearest_create()
knn.train(finalData, cv2.ml.ROW_SAMPLE, finalLabels)

og_test = original_list[-1]
gt_test = gt_list[-1]

og_img_test = cv2.imread(og_test)
gt_img_test = cv2.imread(gt_test)

classified = og_img_test.copy()

Num_a_processar = og_img_test.shape[0] * og_img_test.shape[1]
Num_processado = 0
start_time = time.time()
i = 5
for w in range(og_img_test.shape[0]):
    for h in range(og_img_test.shape[1]):
        if i % 5 != 0:
            Num_processado += 1
            classified[w][h][:] = (0,0,0)
            i += 1
            continue

        cor = og_img_test[w][h][:]
        cor = np.expand_dims(cor, axis=1)
        cor = cor.reshape((-1,3))
        #print("cor shape = ", cor.shape)
        label = classify(knn, np.float32(cor))
        if label == 0:
            classified[w][h][:] = (0,0,0)
        else:
            classified[w][h][:] = (255,255,255)

        if Num_processado == 0 or Num_processado % 3900 == 0:
            print("Passo demorou: {:.2f}seg.\tJa foram processados {} pontos de um total de {} pontos ===> {:.2f}%".format(time.time() - start_time,Num_processado, Num_a_processar, (Num_processado/Num_a_processar)*100))
            start_time = time.time()
        Num_processado +=1
        i += 1

cv2.imshow("imagem original", og_img_test)
cv2.imshow("Ground Truth", gt_img_test)
cv2.imshow("Classified", classified)
cv2.waitKey(0)

'''
import numpy as np
import cv2

img = cv2.imread('home.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''