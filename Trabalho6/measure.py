import numpy as np

def Jaccard(bbox_gt, bbox_medido):
    '''
    Recebe o bounding box do ground truth e o bounding box resultado do tracker,
calcula e retorna o indice de Jaccard que eh a intersecao entre os dois bounding boxes
dividida pela uniao entre eles.
    Entrada = [left,top,bottom,right]
    '''
    for value in bbox_gt:
        if np.isnan(value):
            print("[DEBUG] Jaccard = 0 por causa de NAN no GT")
            return 0.0
    for value in bbox_medido:
        if np.isnan(value):
            print("[DEBUG] Jaccard = 0 por causa de NAN na Medicao")
            return 0.0

    x1_GT, y1_GT, x2_GT, y2_GT = bbox_gt
    x1_med, y1_med, x2_med, y2_med = bbox_medido

    #Calcula a area medida
    Area_med = abs(x1_med - x2_med) * abs(y1_med - y2_med)
    Area_GT = abs(x1_GT - x2_GT) * abs(y1_GT - y2_GT)

    if (max(x1_med, x1_GT) > min(x2_GT, x2_med)) or (max(y1_GT, y1_med) > min(y2_GT, y2_med)):
        Area_intersecao = 0
    else:
        Area_intersecao = abs(max(x1_med, x1_GT) - min(x2_GT, x2_med)) * abs(max(y1_GT, y1_med) - min(y2_GT, y2_med))

    Area_Uniao = Area_med + Area_GT - Area_intersecao

    JaccardIndex = Area_intersecao / Area_Uniao

    print("[DEBUG] Jaccard = {}".format(JaccardIndex))

    return JaccardIndex

def robustez (F, N):
    '''
    Calcula a robustez do tracker. recebe o numero de falhas e o numero total de frames da sequencia
    '''
    S = 30
    M = F / N

    robustez = np.exp(-(S*M))

    return robustez