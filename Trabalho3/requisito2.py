import cv2
import numpy as np 

cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)

# initialize the keypoint detector
detector = cv2.ORB_create(nfeatures = 500)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)


while True:
    _, frame1 = cam1.read()
    _, frame2 = cam2.read()

    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # detect keypoints as well as extract local invariant descriptors
    (kps2, des2) = detector.detectAndCompute(frame2, None)

    #cv2.drawKeypoints(frame2,kps2, frame2, color=(0,255,0), flags=0)

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # detect keypoints as well as extract local invariant descriptors
    (kps1, des1) = detector.detectAndCompute(frame1, None)

    #cv2.drawKeypoints(frame1,kps1, frame1, color=(0,255,0), flags=0)

    cv2.imshow("Note", frame1)
    cv2.imshow("Web", frame2)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
'''
#USING Brute Force match method

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

image = np.zeros((frame1.shape[0], frame1.shape[1]))
print("MATCHES = ", len(matches))

# Draw first 10 matches.
image = cv2.drawMatches(frame1,kps1,frame2,kps2,matches[:10], image, flags=2)
'''


#USING FLANN match method

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(np.asarray(des1,np.float32), np.asarray(des2,np.float32) ,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 2)

image = cv2.drawMatchesKnn(frame1,kps1,frame2,kps2,matches,None,**draw_params)


good = []
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kps2[m.trainIdx].pt)
        pts1.append(kps1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(frame1,frame2,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(frame2,frame1,lines2,pts2,pts1)


epilines = np.hstack((img3, img5))
epilines = cv2.resize(epilines, (int(epilines.shape[1]), int(epilines.shape[0])))
cv2.imshow("epilines", epilines)

print("Fundamental Matrix = ", F)

_, H1, H2 = cv2.stereoRectifyUncalibrated(kps1, kps2, F, (frame1.shape[1], frame1.shape[0]))

print("Homography 1 = ", H1)
print("Homography 2 = ", H2)

#cv2.imshow("Result", image)
cv2.waitKey(0)
# show the shape of the keypoints and local invariant descriptors array
print("[INFO] # of keypoints detected: {}".format(len(kps1)))
print("[INFO] feature vector shape: {}".format(des1.shape))