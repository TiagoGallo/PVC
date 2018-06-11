import cv2
import argparse
from imutils import paths
import numpy as np
from kalman import KalmanBoxTracker
from datasetmanager import get_ground_truth
import time
from measure import Jaccard, robustez
import sys

#TODO: Adicionar metodos para medir a qualidade dos trackers
#TODO: Variar alguns parametros do Kalman para dar menos peso pro "sensor"

def main(args):

    bounding_boxes_GT, imagesList = get_ground_truth(args["dataset"])

    tracker = get_tracker(args["tracker"])

    if args["kalman"]:
        useKalman = True
        print("[INFO] Usando Kalman filter")
    else:
        useKalman = False
        print("[INFO] Nao esta usando o kalman filter")

    #Inicializa o contador de falhas e lista contendo os valores de Jaccard para cada frame
    F = 0
    Jac_values = []

    print("[INFO] Aperte 'q' para sair")
    for i, imagePath in enumerate(imagesList):
        img = cv2.imread(imagePath)
        bb = bounding_boxes_GT[i]

        if i == 0:
            # Initialize tracker with first frame and bounding box
            ok = tracker.init(img, (bb[0], bb[1], bb[2] - bb[0], bb[3]-bb[1]))
            print("[DEBUG] Iniciando o tracker na posicao ", (bb[0], bb[1], bb[2] - bb[0], bb[3]-bb[1]))
            if useKalman:
                kalman = KalmanBoxTracker([bb[0], bb[1], bb[2], bb[3]])

        else:
            # Update tracker
            ok, bbox = tracker.update(img)
           
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                
                if not useKalman:
                    cv2.rectangle(img, p1, p2, (0,255,0), 2, 1)
                    jac = Jaccard(bb, [p1[0], p1[1], p2[0], p2[1]])
                    Jac_values.append(jac)
                    if jac < 0.1:
                        F += 1 
                        tracker = get_tracker(args["tracker"])
                        print("[DEBUG] Iniciando o tracker na posicao ", (bb[0], bb[1], bb[2] - bb[0], bb[3]-bb[1]))
                        if (np.isnan(bb[0])) or (np.isnan(bb[1])) or (np.isnan(bb[2])) or (np.isnan(bb[3])):
                            print("[DEBUG] Vai reiniciar em 0 pq era nan")
                            bb_init = (0,0,1,1)
                        else:
                            bb_init = (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1])
                        try:
                            ok = tracker.init(img, bb_init)
                        except:
                            e = sys.exc_info()[0]
                            print(e)
                        print("[DEBUG] Reiniciou o tracekr por causa de jac pequeno {}".format(F))
                        print("[DEBUG] Iniciando o tracker na posicao ", (bb[0], bb[1], bb[2] - bb[0], bb[3]-bb[1]))

                if useKalman:
                    bbox_kalman = kalman.predict()
                    cv2.rectangle(img, (int(bbox_kalman[0][0]), int(bbox_kalman[0][1])), (int(bbox_kalman[0][2]), int(bbox_kalman[0][3])), (0,255,0), 2, 1)

                    kalman.update([p1[0],p1[1], p2[0], p2[1]])

                    jac = Jaccard(bb, bbox_kalman)
                    Jac_values.append(jac)
                    if jac < 0.1:
                        F += 1 
                        tracker = get_tracker(args["tracker"])
                        print("[DEBUG] Iniciando o tracker na posicao ", (bb[0], bb[1], bb[2] - bb[0], bb[3]-bb[1]))
                        if (np.isnan(bb[0])) or (np.isnan(bb[1])) or (np.isnan(bb[2])) or (np.isnan(bb[3])):
                            print("[DEBUG] Vai reiniciar em 0 pq era nan")
                            bb_init = (0,0,1,1)
                        else:
                            bb_init = (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1])
                        try:
                            ok = tracker.init(img, bb_init)
                        except:
                            e = sys.exc_info()[0]
                            print(e)
                        print("[DEBUG] Reiniciou o tracekr por causa de jac pequeno {}".format(F))
                        print("[DEBUG] Iniciando o tracker na posicao ", (bb[0], bb[1], bb[2] - bb[0], bb[3]-bb[1]))

            else :
                # Tracking failure
                if useKalman:
                    bbox_kalman = kalman.predict()
                    cv2.rectangle(img, (int(bbox_kalman[0][0]), int(bbox_kalman[0][1])), (int(bbox_kalman[0][2]), int(bbox_kalman[0][3])), (0,0,255), 2, 1)
                    
                    jac = Jaccard(bb, bbox_kalman)
                    Jac_values.append(jac)
                    
                    if jac < 0.1:
                        F += 1 
                        tracker = get_tracker(args["tracker"])
                        print("[DEBUG] Iniciando o tracker na posicao ", (bb[0], bb[1], bb[2] - bb[0], bb[3]-bb[1]))
                        if (np.isnan(bb[0])) or (np.isnan(bb[1])) or (np.isnan(bb[2])) or (np.isnan(bb[3])):
                            print("[DEBUG] Vai reiniciar em 0 pq era nan")
                            bb_init = (0,0,1,1)
                        else:
                            bb_init = (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1])
                        try:
                            ok = tracker.init(img, bb_init)
                        except:
                            e = sys.exc_info()[0]
                            print(e)
                        print("[DEBUG] Reiniciou o tracekr por causa de jac pequeno {}".format(F))
                        print("[DEBUG] Iniciando o tracker na posicao ", (bb[0], bb[1], bb[2] - bb[0], bb[3]-bb[1]))
                else:
                    F += 1
                    tracker = get_tracker(args["tracker"])
                    print("[DEBUG] Iniciando o tracker na posicao ", (bb[0], bb[1], bb[2] - bb[0], bb[3]-bb[1]))
                    if (np.isnan(bb[0])) or (np.isnan(bb[1])) or (np.isnan(bb[2])) or (np.isnan(bb[3])):
                        print("[DEBUG] Vai reiniciar em 0 pq era nan")
                        bb_init = (0,0,1,1)
                    else:
                        bb_init = (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1])
                    try:
                        ok = tracker.init(img, bb_init)
                    except:
                        e = sys.exc_info()[0]
                        print(e)
                    print("[DEBUG] Reiniciou o tracekr pq o tracker se perdeu {}".format(F))

                #TODO: Reiniciar o tracekr 
                

        #Draw ground Truth when it exists
        if not np.isnan(bb[0]):
            cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255,0,0), 2, 1)

        #Desenha legenda do grafico
        cv2.line(img, (5, 10), (20, 10), (255,0,0), 2)
        cv2.putText(img, "Ground Truth", (22, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
        cv2.line(img, (5, 30), (20, 30), (0,255,0), 2)
        if useKalman:
            cv2.putText(img, "Tracker ({} + kalman)".format(args["tracker"]), (22, 35), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
        else:
            cv2.putText(img, "Tracker ({})".format(args["tracker"]), (22, 35), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)

        cv2.imshow("tracker", img)

        k = cv2.waitKey(1)

        if k == ord("q"):
            break

    #Calcula com 1 frame a menos que a quantidade de frames, pois o primeiro eh usado so pra inicializar o tracker
    Robustness = robustez(F, len(imagesList) - 1)
    Med_Jac = sum(Jac_values)/len(Jac_values)

    print("[INFO] A robustez foi de {}\nA media do Jaccard foi {}".format(Robustness, Med_Jac))


def get_tracker(tracker_name):
    '''
    Recebe o nome do tracker que sera utilizado e retorna um objeto do OpenCV 'contendo' o tracker
    '''
    if tracker_name == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_name == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_name == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_name == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_name == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    else:
        raise NameError ("O tracker {} nao eh suportado".format(tracker_name))

    return tracker

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("-t", "--tracker", default="BOOSTING",
        help="OpenCV tracker name\nPossible names: BOOSTING"
        ", MIL, TLD, KCF, MEDIANFLOW")
    ap.add_argument("-k", "--kalman", default=False,
        help="Set true if want to use Kalman filter")
    ap.add_argument("-d", "--dataset", default="Professor_car1",
        help="Which dataset to use\nPossible names: car1,"
        " car2")
    args = vars(ap.parse_args())
    
    
    main(args)