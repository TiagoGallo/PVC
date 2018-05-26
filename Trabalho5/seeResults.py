from imutils import paths
from random import shuffle
import cv2
from keras.preprocessing.image import img_to_array
import os
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import argparse
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from utils import get_inception_with_frozen_layers

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True,
    help="path to output model")
args = vars(ap.parse_args())

model = get_inception_with_frozen_layers(102)
model.load_weights(args["model"])

imgsList = sorted(list(paths.list_images(args["dataset"])))
shuffle(imgsList)

print("Aperte q para sair")

for imagePath in imgsList:
    data = []
    labels = []
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200,200))
    mostrar = image.copy()
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    le = LabelEncoder().fit(labels)
    labels = np_utils.to_categorical(le.transform(labels), 102)

    predictions = model.predict(data).argmax(axis=1)
    lab = labels.argmax(axis=1)
    #print("pred foi = {}\nlab = {}\ntarget name = {}".format(predictions, lab, le.classes_))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(mostrar,le.classes_[0],(5,180), font, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow("image", mostrar)
    k = cv2.waitKey(0)

    if k == ord('q'):
        break
