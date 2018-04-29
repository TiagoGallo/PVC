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

''''
USING Brute Force match method

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

'''
USING FLANN match method

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
'''

cv2.imshow("Result", image)
cv2.waitKey(0)
# show the shape of the keypoints and local invariant descriptors array
print("[INFO] # of keypoints detected: {}".format(len(kps1)))
print("[INFO] feature vector shape: {}".format(des1.shape))