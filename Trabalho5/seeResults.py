import imutils
import cv2
from keras.preprocessing.image import img_to_array,array_to_img
import numpy as np
import argparse
from utils import get_inception_with_frozen_layers, load_images_to_memory

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True,
    help="path to output model")
args = vars(ap.parse_args())

model = get_inception_with_frozen_layers(102)
model.load_weights(args["model"])

le, data, labels = load_images_to_memory(args["dataset"], randomOrder=True)

font = cv2.FONT_HERSHEY_SIMPLEX

print("Aperte 'q' para sair!!")
for dat, label in zip(data, labels):
    dats = np.expand_dims(dat, axis=0)
    pred = model.predict(dats)
    #print ("pred argmax = {}\t label = {}\t le classes = {}".format(pred.argmax(axis=1), label.argmax(axis=0), le.classes_[pred.argmax(axis=1)[0]]))

    predito = pred.argmax(axis=1)[0]
    verdade = label.argmax(axis=0)
    classe = le.classes_[predito]

    dat = array_to_img(dat)
    open_cv_image = np.array(dat) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 

    open_cv_image = imutils.resize(open_cv_image, width=400)

    if predito == verdade:
        cv2.putText(open_cv_image,'{}: {:.2f}%'.format(classe, pred[0][predito] * 100),(2,30), font, 1,(0,255,0),2,cv2.LINE_AA)
    else:
        cv2.putText(open_cv_image,'{}: {:.2f}%'.format(classe, pred[0][predito] * 100),(2,30), font, 1,(0,0,255),2,cv2.LINE_AA)

    cv2.imshow("teste", open_cv_image)
    k = cv2.waitKey(0)

    if k == ord('q'):
        break
