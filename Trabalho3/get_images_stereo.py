import cv2 
import numpy as np

vidL = cv2.VideoCapture(1)
vidR = cv2.VideoCapture(0)

termination_criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# set up a set of real-world "object points" for the chessboard pattern

patternX = 8
patternY = 6 
square_size_in_mm = 40 

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((patternX*patternY,3), np.float32)
objp[:,:2] = np.mgrid[0:patternX,0:patternY].T.reshape(-1,2)
objp = objp * square_size_in_mm 

# create arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsR = [] # 2d points in image plane.
imgpointsL = [] # 2d points in image plane.

# count number of chessboard detection (across both images)
chessboard_pattern_detections = 0 

countImages = 0
while countImages != 20:
    _, frameL = vidL.read()
    _, frameR = vidR.read()

    frameLeft = frameL.copy()
    frameRight = frameR.copy()

    grayL = cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY) 
    grayR = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY) 

    # Find the chess board corners in the image
    # (change flags to perhaps improve detection ?)

    retR, cornersL = cv2.findChessboardCorners(grayL, (patternX,patternY),None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE) 
    retL, cornersR = cv2.findChessboardCorners(grayR, (patternX,patternY),None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE) 

    # If found, add object points, image points (after refining them)

    if ((retR == True) and (retL == True)):
        # add object points to global list

        objpoints.append(objp) 

        # refine corner locations to sub-pixel accuracy and then

        corners_sp_L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),termination_criteria_subpix) 
        imgpointsL.append(corners_sp_L) 
        corners_sp_R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),termination_criteria_subpix) 
        imgpointsR.append(corners_sp_R) 

        # Draw and display the corners

        drawboardL = cv2.drawChessboardCorners(frameL, (patternX,patternY), corners_sp_L,retL)
        drawboardR = cv2.drawChessboardCorners(frameR, (patternX,patternY), corners_sp_R,retR)

    cv2.imshow("left", frameL)
    cv2.imshow("right", frameR)

    k = cv2.waitKey(1)
    if k == ord('r') and ((retR == True) and (retL == True)):
        countImages += 1
        cv2.imwrite("./calibrationImages/calibrateLeft{}.png".format(countImages), frameLeft)
        cv2.imwrite("./calibrationImages/calibrateRight{}.png".format(countImages), frameRight)
        print("Salva imagem de calibracao numero ", countImages)

print ("Salvas todas as imagens de calibracao")

countImages = 0
while countImages != 5:
    _, frameL = vidL.read()
    _, frameR = vidR.read()

    cv2.imshow("left", frameL)
    cv2.imshow("right", frameR)

    k = cv2.waitKey(1)
    if k == ord('r'):
        countImages += 1
        cv2.imwrite("./testeImages/testeLeft{}.png".format(countImages), frameL)
        cv2.imwrite("./testeImages/testeRight{}.png".format(countImages), frameR)
        print("Salva imagem de teste numero ", countImages)