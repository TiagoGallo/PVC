from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from random import shuffle

def load_images_to_memory(path, randomOrder=False):
    data = []
    labels = []

    imgsList = sorted(list(paths.list_images(path)))

    if randomOrder:
        shuffle(imgsList)

    # loop over the input images
    for imagePath in imgsList:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200,200))
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

    return le, data, labels

def get_inception_with_frozen_layers(num_classes):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.5)(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    return model

def train_model(model, testX, trainX, testY, trainY, n_epochs):
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the network
    print("[INFO] warming the network head...")
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
        batch_size=128, epochs=15, verbose=1)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    # construct the callback to save only the *best* model to disk
    # based on the validation loss
    checkpoint = ModelCheckpoint("./weights/InceptionV3.h5", monitor="val_loss",
            save_best_only=True, verbose=1)
    callbacks = [checkpoint]

    # train the network
    print("[INFO] training network...")
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
        batch_size=204, epochs=n_epochs, verbose=1, callbacks=callbacks)

    return model, H

def evaluate_model(model, testY, testX, le):
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=64)
    print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=le.classes_))

def plot_val_acc(H, n_epochs):
    # plot the training + testing loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, n_epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n_epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, n_epochs), H.history["acc"], label="acc")
    plt.plot(np.arange(0, n_epochs), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()