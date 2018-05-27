import os
from shutil import copyfile, move
from imutils import paths

directory = './CalTech101/Test'

dirs = [x[0] for x in os.walk(directory)]

for dir in dirs:
    if dir.split(os.path.sep)[-1] == 'Test':
        continue

      
    img21 = os.path.join(dir, 'image_0021.jpg')
    img22 = os.path.join(dir, 'image_0022.jpg')
    img23 = os.path.join(dir, 'image_0023.jpg')
    img24 = os.path.join(dir, 'image_0024.jpg')
    img25 = os.path.join(dir, 'image_0025.jpg')
    img26 = os.path.join(dir, 'image_0026.jpg')
    img27 = os.path.join(dir, 'image_0027.jpg')
    img28 = os.path.join(dir, 'image_0028.jpg')
    img29 = os.path.join(dir, 'image_0029.jpg')
    img30 = os.path.join(dir, 'image_0030.jpg')

    imgDest21 = img21.replace('Test', 'Train')
    imgDest22 = img22.replace('Test', 'Train')
    imgDest23 = img23.replace('Test', 'Train')
    imgDest24 = img24.replace('Test', 'Train')
    imgDest25 = img25.replace('Test', 'Train')
    imgDest26 = img26.replace('Test', 'Train')
    imgDest27 = img27.replace('Test', 'Train')
    imgDest28 = img28.replace('Test', 'Train')
    imgDest29 = img29.replace('Test', 'Train')
    imgDest30 = img30.replace('Test', 'Train')

    move(img21, imgDest21)
    move(img22, imgDest22)
    move(img23, imgDest23)
    move(img24, imgDest24)
    move(img25, imgDest25)
    move(img26, imgDest26)
    move(img27, imgDest27)
    move(img28, imgDest28)
    move(img29, imgDest29)
    move(img30, imgDest30)