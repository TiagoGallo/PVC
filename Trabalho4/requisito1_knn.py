import numpy as np
import cv2
import os

def classify(centroPele, centroFundo, Ponto):
    distPele = ((Ponto[0] - centroPele[0])**2 + (Ponto[1] - centroPele[1])**2 + (Ponto[2] - centroPele[2])**2) ** (1/2)
    distFundo = ((Ponto[0] - centroFundo[0])**2 + (Ponto[1] - centroFundo[1])**2 + (Ponto[2] - centroFundo[2])**2) ** (1/2)

    if distPele < distFundo:
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
for og_image, gt_image in zip(original_list, gt_list):
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

    Z = np.hstack([Z_og, list_for_gt])
    #print("final data shape = ", Z.shape)

    if imgNumber == 0:
        finalData = Z
    else:
        finalData = np.vstack([finalData, Z])

    #cv2.imshow("original", img_og)
    #cv2.imshow("groundTruth", img_gt)
    #print("Tiveram {} pixels de fundo e {} pixels de cor de pele".format(fundo, pele))
    #cv2.waitKey(0)
    imgNumber += 1


#print ("FinalData shape = ", finalData.shape)
finalData = np.float32(finalData)


# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(finalData,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

print ("centers = ", center)

centerFundo = center[0]
centerPele = center[1]

centerFundo = centerFundo[:-1]
centerPele = centerPele[:-1]

print("Centro da pele = {}\nCentro do fundo = {}".format(centerPele, centerFundo))


og_test = original_list[0]
gt_test = gt_list[0]

og_img_test = cv2.imread(og_test)
gt_img_test = cv2.imread(gt_test)

classified = og_img_test.copy()

for w in range(og_img_test.shape[0]):
    for h in range(og_img_test.shape[1]):
        cor = og_img_test[w][h][:]
        label = classify(centerPele, centerFundo, cor)
        if label == 0:
            classified[w][h][:] = (255,255,255)
        else:
            classified[w][h][:] = (0,0,0)

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