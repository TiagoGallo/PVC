import numpy as np
import cv2

f = 25
baseLine = 120

imgL = cv2.imread('./imgs-estereo/aloeL.png',0)
imgR = cv2.imread('./imgs-estereo/aloeR.png',0)

#imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
#imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM().create(numDisparities=112, blockSize=17)
disparity = stereo.compute(imgL,imgR).astype(np.float32) / 16.0

size = disparity.shape

depth = np.zeros((size[0], size[1], 1))

#greater_depth = 0
#lesser_depth = 100000000

for i in range(0, size[0]):
    for j in range(0, size[1]):
        if disparity[i][j] != 0.0:
            depth[i][j] = (baseLine * f) / disparity[i][j]

        #if depth[i][j] > greater_depth:
        #    greater_depth = depth[i][j]
        #if depth[i][j] < lesser_depth:
        #    lesser_depth = depth[i][j]

#print("greater = {}\n lesser = {}".format(greater_depth, lesser_depth))

#NORMALIZE
depth_min = depth.min()
depth_max = depth.max()
depth = depth - depth_min
depth = depth / depth_max

depth = depth * 255
        

r = 3
h_new = int(size[0] / r)
w_new = int(size[1] / r)
disparity = cv2.resize(disparity, (h_new, w_new))
depth = cv2.resize(depth, (h_new, w_new))

#depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

cv2.imshow("haha", disparity)
cv2.imshow("depth", depth)
cv2.waitKey(0)