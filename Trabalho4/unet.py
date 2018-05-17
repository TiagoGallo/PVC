from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD
from keras import backend as K
import tensorflow as tf
import keras
import random 
import os
import numpy as np
import cv2
import sys

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

def train_generator_multiclass(image_generator, mask_generator):
    while True:
        image = next(image_generator)
        mask = next(mask_generator)
        label = make_regressor_label(mask).astype(np.float32)
        yield (image, label)

def make_regressor_label(gt):
    human = np.where(gt==24,255,0) + np.where(gt==25,255,0)
    car = np.where(gt==26,255,0) + np.where(gt==27,255,0) + np.where(gt==28,20,0)
    road = np.where(gt==7,255,0) #+ np.where(gt==8,1,0)
    label = np.concatenate((human, car, road), axis=-1)
    return label

def train_generator(image_generator, mask_generator):
    while True:
        yield(next(image_generator), next(mask_generator))

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true /= 255.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def pixelwise_binary_ce(y_true, y_pred):
    y_true /= 255.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.mean(K.binary_crossentropy(y_pred_f, y_true_f))

def get_unet_1class(image_height, image_width):
    img_rows = image_height
    img_cols = image_width
    lr = 0.005
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation=None, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(32, (3, 3), activation=None, padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation=None, padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(64, (3, 3), activation=None, padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation=None, padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(128, (3, 3), activation=None, padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation=None, padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(256, (3, 3), activation=None, padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation=None, padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(512, (3, 3), activation=None, padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation=None, padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(256, (3, 3), activation=None, padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation=None, padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(128, (3, 3), activation=None, padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation=None, padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(64, (3, 3), activation=None, padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation=None, padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(32, (3, 3), activation=None, padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=pixelwise_binary_ce, metrics=[dice_coef, 'accuracy'])

    return model

train_mode = False
visualize = True

batch_size = 2
data_path = './dataset/big/'

image_height = 768
image_width = 512

if train_mode:
    datagen_args = dict(featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
                    # fill_mode='constant',
                    # cval=0.,
                    horizontal_flip=False,  # randomly flip images
                    vertical_flip=False,
                    validation_split=0.2)  # randomly flip images

    image_datagen = ImageDataGenerator(**datagen_args)
    mask_datagen = ImageDataGenerator(**datagen_args)

    ### generator
    seed = random.randrange(1, 1000)
    image_generator = image_datagen.flow_from_directory(
                os.path.join(data_path, 'train/image'),
                class_mode=None, seed=seed, batch_size=batch_size, 
                target_size=(image_height, image_width),
                color_mode='grayscale')
    mask_generator = mask_datagen.flow_from_directory(
                os.path.join(data_path, 'train/gt'),
                class_mode=None, seed=seed, batch_size=batch_size, 
                target_size=(image_height, image_width),
                color_mode='grayscale')

    model = get_unet_1class(image_height, image_width)
    model.summary()

    ### train model
    model.fit_generator(
        train_generator(image_generator, mask_generator),
        #train_generator_multiclass(image_generator, mask_generator),
        steps_per_epoch= image_generator.n // batch_size,
        epochs=20
    )

    model.save_weights('./weights/peso3.h5')

else:
    
    testImages = create_imageList_fromDirectory('./dataset/big/test/image/ori/')

    resized = False
    model = get_unet_1class(image_height, image_width)
    model.load_weights('./weights/peso3.h5', by_name=False)

    for imagePath in testImages:

        bgr_img = cv2.imread(imagePath, 1)
        if bgr_img.shape != (768,512,3):
            resized = True
            bgr_img= cv2.resize(bgr_img,(512,768))

        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        input_data = rgb_img[None,:,:]

        #print("shape = ", input_data.shape)

        input_data = np.expand_dims(input_data, axis=3)

        #print("shape = ", input_data.shape)

        result = model.predict(input_data)#, 1)

        param = np.max(result[0]) / 17
        for i in range(result[0].shape[0]):
            for j in range(result[0].shape[1]):
                if result[0][i][j] < param:
                    result[0][i][j] = 0
                else:
                    result[0][i][j] = 255

        imgMask = (result[0]).astype(np.uint8)

        if resized:
            imgMask = cv2.resize(imgMask, (768,512))
        
        #print("Nome de saida = ", './dataset/big/masked/' + imagePath.split('/')[-1])

        imagepathout = './dataset/big/masked/' + imagePath.split('/')[-1]
        cv2.imwrite(imagepathout, imgMask)

        if visualize:
            print("Aperte qualquer tecla para passar pra proxima imagem ou 'q' para sair")
            cv2.imshow('mask' ,imgMask)
            k = cv2.waitKey(0)

            if k == ord('q'):
                sys.exit()


'''
print("Antes da exponencial: maximo = {}     minimo = {}".format(np.max(result[0]), np.min(result[0])))

    result[0] = np.power(result[0], (1/np.e))
    maximo = np.max(result[0])
    minimo = np.min(result[0])

    result[0] = result[0] - minimo
    result[0] = result[0] / maximo

    print("maximo inicial = {}    minimo inicial = {}\nMaximo modificado = {}    minimo modificado = {}".format(maximo, minimo, np.max(result[0]), np.min(result[0])))

    imgMask = (result[0]*255).astype(np.uint8)

    imgMask[imgMask>45] = 255
    imgMask[imgMask<=45] = 0
'''

'''
 param = np.max(result[0]) / 20
    for i in range(result[0].shape[0]):
        for j in range(result[0].shape[1]):
            if result[0][i][j] < param:
                result[0][i][j] = 0
            else:
                result[0][i][j] = 255

    imgMask = (result[0]).astype(np.uint8)
    if resized:
        imgMask = cv2.resize(imgMask, (768,512))
    cv2.imshow('mask' ,imgMask)
    cv2.waitKey(0)
'''