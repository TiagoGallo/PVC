import cv2
import os

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

def JaccardIndex(imageFile, GroundTruthFile):
    maskImage = cv2.imread(imageFile)
    GTimage = cv2.imread(GroundTruthFile)

    intersectionPoints = 0
    unionPoints = 0

    for w in range(maskImage.shape[0]):
        for h in range(maskImage.shape[1]):
            
            cor_mask = maskImage[w][h][:]
            cor_GT = GTimage[w][h][:]

            if ((int(cor_mask[0]) == int(cor_mask[1]) == int(cor_mask[2]) == 255) and (int(cor_GT[0]) + int(cor_GT[1]) + int(cor_GT[2]) != 0)):     #mascara branca e pixel nao preto no gt
                intersectionPoints += 1
                unionPoints += 1
            elif((int(cor_mask[0]) == int(cor_mask[1]) == int(cor_mask[2]) == 255) or (int(cor_GT[0]) + int(cor_GT[1]) + int(cor_GT[2]) != 0)):     #Apenas um dos dois eh true para o pixel
                unionPoints += 1

    return (intersectionPoints / unionPoints)

def BasicAccuracy(imageFile, GroundTruthFile):
    maskImage = cv2.imread(imageFile)
    GTimage = cv2.imread(GroundTruthFile)

    pixelsCertos = 0
    pixelsTotal = maskImage.shape[0] * maskImage.shape[1]

    for w in range(maskImage.shape[0]):
        for h in range(maskImage.shape[1]):
            
            cor_mask = maskImage[w][h][:]
            cor_GT = GTimage[w][h][:]
            
            if (((int(cor_mask[0]) == int(cor_mask[1]) == int(cor_mask[2]) == 255) and (int(cor_GT[0]) + int(cor_GT[1]) + int(cor_GT[2]) != 0)) or  #mascara branca e pixel nao preto no gt
                ((int(cor_mask[0]) == int(cor_mask[1]) == int(cor_mask[2]) ==  0 ) and (int(cor_GT[0]) + int(cor_GT[1]) + int(cor_GT[2]) == 0))):   #mascara preta e pixel preto no gt
            
                pixelsCertos += 1

    return (pixelsCertos / pixelsTotal)


if __name__ == '__main__':
    imgPath = './SkinDataset/Masked/knn/'
    imagesPaths = create_imageList_fromDirectory(imgPath)

    maskPath = './SkinDataset/GT/test/'
    maskPaths = create_imageList_fromDirectory(maskPath)

    Jac_acc = 0
    Bas_acc = 0

    print("Total de imgs = ", len(maskPaths))
    i = 0
    for image, maks in zip(imagesPaths, maskPaths):
        Jac_acc += JaccardIndex(image, maks)
        Bas_acc += BasicAccuracy(image, maks)

        i += 1
        print("Processando imagem {}\n image = {} \n mask = {}\nA acuracia foi de {:.2f}% usando Jaccard index e {:.2f}% usando acuracia basica".format(i, image, maks, Jac_acc * 100, Bas_acc * 100))

    Jac_acc /= len(imagesPaths)
    Bas_acc /= len(imagesPaths)


    print("A acuracia foi de {:.2f}% usando Jaccard index e {:.2f}% usando acuracia basica".format(Jac_acc * 100, Bas_acc * 100))