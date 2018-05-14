import cv2

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
    imageFile = './SkinDataset/Masked/knn/278.jpg'
    GroundTruthFile = './SkinDataset/GT/test/278.jpg'

    Jac_acc = JaccardIndex(imageFile, GroundTruthFile)
    Bas_acc = BasicAccuracy(imageFile, GroundTruthFile)

    print("A acuracia foi de {:.2f}% usando Jaccard index e {:.2f}% usando acuracia basica".format(Jac_acc * 100, Bas_acc * 100))