import cv2
import argparse
from imutils import paths
import numpy as np
from kalman import KalmanBoxTracker

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

    print("[INFO] Aperte 'q' para sair")
    for i, imagePath in enumerate(imagesList):
        img = cv2.imread(imagePath)
        bb = bounding_boxes_GT[i]

        if i == 0:
            # Initialize tracker with first frame and bounding box
            ok = tracker.init(img, (bb[0], bb[1], bb[2] - bb[0], bb[3]-bb[1]))

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

                if useKalman:
                    bbox_kalman = kalman.predict()
                    cv2.rectangle(img, (int(bbox_kalman[0][0]), int(bbox_kalman[0][1])), (int(bbox_kalman[0][2]), int(bbox_kalman[0][3])), (0,255,0), 2, 1)

                    kalman.update([p1[0],p1[1], p2[0], p2[1]])

            else :
                # Tracking failure
                if useKalman:
                    bbox_kalman = kalman.predict()
                    cv2.rectangle(img, (int(bbox_kalman[0][0]), int(bbox_kalman[0][1])), (int(bbox_kalman[0][2]), int(bbox_kalman[0][3])), (0,0,255), 2, 1)

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


def get_ground_truth(dataset):
    '''
    Recebe o nome do dataset a ser utilizado e retorna uma lista com os bounding boxes de ground truth
    e uma lista com as imagens referentes a esse dataset
    '''
    if dataset == "car1":
        filename = './data/gtcar1.txt'
        imagesPath = './data/car1'
    elif dataset == "car2":
        filename = './data/gtcar2.txt'
        imagesPath = './data/car2'
    else:
        raise NameError ("There is no {} dataset".format(dataset))

    #Create a list to store the ground truth bounding boxes
    bounding_boxes = []

    with open(filename) as f:
        data = f.readlines()

    for i in range(len(data)):
        data[i] = data[i].replace("\n", "")
        top, left, bottom, right = data[i].split(",")
        bb = [float(top), float(left), float(bottom), float(right)]
        bounding_boxes.append(bb)

    imagesList = sorted(list(paths.list_images(imagesPath)))

    return bounding_boxes, imagesList

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
    ap.add_argument("-d", "--dataset", default="car1",
        help="Which dataset to use\nPossible names: car1,"
        " car2")
    args = vars(ap.parse_args())
    
    
    main(args)