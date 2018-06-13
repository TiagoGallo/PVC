import cv2
import argparse
from imutils import paths
import numpy as np
from datasetmanager import get_ground_truth
import time
from measure import Jaccard, robustez
import sys


def main(args):
    print()
    bounding_boxes_GT, imagesList = get_ground_truth(args["dataset"])

    tracker = get_tracker(args["tracker"])

    #Inicializa o contador de falhas e lista contendo os valores de Jaccard para cada frame
    F = 0
    Jac_values = []

    print("Tracker = {}\t Dataset = {}".format(args["tracker"], args["dataset"]))

    #print("[INFO] Aperte 'q' para sair")
    for i, imagePath in enumerate(imagesList):
        img = cv2.imread(imagePath)
        
        bb = bounding_boxes_GT[i]
        bb_unchanged = bb.copy()

        if i == 0:
            # Initialize tracker with first frame and bounding box
            ok = tracker.init(img, (bb[0], bb[1], bb[2] - bb[0], bb[3]-bb[1]))            

        else:
            # Update tracker
            ok, bbox = tracker.update(img)
           
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                
                cv2.rectangle(img, p1, p2, (0,255,0), 2, 1)
                jac = Jaccard(bb, [p1[0], p1[1], p2[0], p2[1]])
                Jac_values.append(jac)
                if jac < 0.1: 
                    del(tracker)
                    tracker = get_tracker(args["tracker"])                        
                    if (np.isnan(bb[0])) or (np.isnan(bb[1])) or (np.isnan(bb[2])) or (np.isnan(bb[3])):
                        #print("[DEBUG] Vai reiniciar em 0 pq era nan")
                        bb[0] = 0
                        bb[1] = 0
                        bb[2] = 10
                        bb[3] = 10
                    else:
                        F += 1
                    if (bb[0] < 0) or (bb[1] < 0) or (bb[2] < 0) or (bb[3] < 0):
                        #print("[DEBUG] Vai reiniciar em 0 pq era negativo = ", bb)
                        if bb[0] < 0.0: bb[0] = 0 
                        if bb[1] < 0.0: bb[1] = 0
                        #print("[DEBUG] Novo bb = ", bb)
                    
                    ok = tracker.init(img, (bb[0], bb[1], bb[2] - bb[0], bb[3]-bb[1]))
                                                
                                                

            else :
                del(tracker)
                tracker = get_tracker(args["tracker"])                    
                if (np.isnan(bb[0])) or (np.isnan(bb[1])) or (np.isnan(bb[2])) or (np.isnan(bb[3])):
                        #print("[DEBUG] Vai reiniciar em 0 pq era nan")
                        bb[0] = 0
                        bb[1] = 0
                        bb[2] = 10
                        bb[3] = 10
                else:
                    F += 1
                if (bb[0] < 0) or (bb[1] < 0) or (bb[2] < 0) or (bb[3] < 0):
                    #print("[DEBUG] Vai reiniciar em 0 pq era negativo = ", bb)
                    if bb[0] < 0.0: bb[0] = 0 
                    if bb[1] < 0.0: bb[1] = 0
                    #print("[DEBUG] Novo bb = ", bb)
                
                ok = tracker.init(img, (bb[0], bb[1], bb[2] - bb[0], bb[3]-bb[1]))

                    #print("[DEBUG] Reiniciou o tracekr pq o tracker se perdeu {}".format(F))

                

        #Draw ground Truth when it exists
        if not np.isnan(bb_unchanged[0]):
            cv2.rectangle(img, (int(bb_unchanged[0]), int(bb_unchanged[1])), (int(bb_unchanged[2]), int(bb_unchanged[3])), (255,0,0), 2, 1)

        #Desenha legenda do grafico
        cv2.line(img, (5, 10), (20, 10), (255,0,0), 2)
        cv2.putText(img, "Ground Truth", (22, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
        cv2.line(img, (5, 30), (20, 30), (0,255,0), 2)
        
        cv2.putText(img, "Tracker ({})".format(args["tracker"]), (22, 35), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)

        cv2.imshow("tracker", img)

        k = cv2.waitKey(1)

        if k == ord("q"):
            break

    #Calcula com 1 frame a menos que a quantidade de frames, pois o primeiro eh usado so pra inicializar o tracker
    Robustness = robustez(F, len(imagesList) - 1)
    Med_Jac = sum(Jac_values)/len(Jac_values)

    print("A robustez foi de {}\nA media do Jaccard foi {}".format(Robustness, Med_Jac))
    print()

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
    ap.add_argument("-d", "--dataset", default="Professor_car1",
        help="Which dataset to use\nPossible names: car1,"
        " car2")
    args = vars(ap.parse_args())
    
    
    main(args)