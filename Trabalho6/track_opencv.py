import cv2
from imutils import paths
import numpy as np
# It is a good idea to store the filename into a variable.
# The variable can later become a function argument when the
# code is converted to a function body.
filename = './data/gtcar1.txt'

# Using the newer with construct to close the file automatically.
with open(filename) as f:
    data = f.readlines()

bounding_boxes = []

for i in range(len(data)):
    data[i] = data[i].replace("\n", "")
    top, left, bottom, right = data[i].split(",")
    bb = [float(top), float(left), float(bottom), float(right)]
    bounding_boxes.append(bb)

imagesList = sorted(list(paths.list_images("./data/car1")))

tracker_type = 'BOOSTING'

if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
elif tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()

i = 0
for imagePath in imagesList:
    img = cv2.imread(imagePath)

    bb = bounding_boxes[i]

    if i == 0:
        # Initialize tracker with first frame and bounding box
        ok = tracker.init(img, (bb[0], bb[1], bb[2] - bb[0], bb[3]-bb[1]))
    else:
        # Update tracker
        ok, bbox = tracker.update(img)
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(img, p1, p2, (0,255,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(img, "Tracking failure detected", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    if not np.isnan(bb[0]):
        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255,0,0), 2, 1)

    cv2.imshow("wow", img)

    k = cv2.waitKey(10)

    if k == ord("q"):
        break

    i += 1