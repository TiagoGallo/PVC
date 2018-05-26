from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
from utils import load_images_to_memory,\
                get_inception_with_frozen_layers,\
                train_model, evaluate_model,\
                plot_val_acc


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input training dataset")
#ap.add_argument("-m", "--model", required=True,
#    help="path to output model")
args = vars(ap.parse_args())

n_epochs = 40


le, data, labels = load_images_to_memory(args["dataset"])

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.20, stratify=labels, random_state=42)

model = get_inception_with_frozen_layers(102)

model, H = train_model(model, testX, trainX, testY, trainY, n_epochs)

evaluate_model(model, testY, testX, le)

plot_val_acc(H, n_epochs)
