# import necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
from datasetmanager import get_ground_truth


class State:
    """
    Class to store state.
    """

    def __init__(self, x, y, vx=None, vy=None):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy


class Measure:
    """
    Class to store measure.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y


def find_object(img_full, img_obj, matcher):
    """
    Finds the object represented by img_obj in img_full using template matching. 
    """

    # template shape info
    h, w = img_obj.shape[0:2]

    # initialize the measure
    pos = None

    # find the object using template matching
    if matcher == 'ccoeff':
        res = cv2.matchTemplate(img_full, img_obj, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        pos = Measure(int(max_loc[0] + w/2), int(max_loc[1] + h/2))
    elif matcher == 'ccorr':
        res = cv2.matchTemplate(img_full, img_obj, cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        pos = Measure(int(max_loc[0] + w/2), int(max_loc[1] + h/2))
    elif matcher == 'sqdiff':
        res = cv2.matchTemplate(img_full, img_obj, cv2.TM_SQDIFF_NORMED)
        min_val, _, min_loc, _ = cv2.minMaxLoc(res)
        pos = Measure(int(min_loc[0] + w/2), int(min_loc[1] + h/2))
    else:
        print('Unknown matcher.')

    # returns the object position
    return pos


def prepare_data(args):
    """
    Prepares the image files list and the annotations list.
    """

    # list all files for the chosen subset
    subset_folder = os.path.join(args['dataset'], 'car{}'.format(args['subset']))
    img_list = sorted([os.path.join(subset_folder, img_file) for img_file in os.listdir(subset_folder) if img_file.endswith('.jpg')])

    # load the annotations for the chosen subset
    annotations_file = open(os.path.join(args['dataset'], 'gtcar{}.txt'.format(args['subset'])), 'r')
    annotations = [[float(n) for n in gt.split(',')] for gt in annotations_file.read().split('\n') if gt != '']
    annotations_file.close()

    # return the data
    return img_list, annotations


def jaccard(a, b):
    """
    Determines the Jaccard index (IoU) for two bounding boxes.
    """

    # determine the range of the boxes
    min_x = np.minimum(a[0], b[0])
    min_y = np.minimum(a[1], b[1])
    max_x = np.maximum(a[2], b[2])
    max_y = np.maximum(a[3], b[3])

    # construct masks
    mask_a = np.zeros((max_y-min_y, max_x-min_x), dtype=int)
    mask_b = np.zeros_like(mask_a)
    mask_a[a[1]-min_y:a[3]-min_y, a[0]-min_x:a[2]-min_x] = 255
    mask_b[b[1]-min_y:b[3]-min_y, b[0]-min_x:b[2]-min_x] = 255

    # get the IoU
    i = np.bitwise_and(mask_a, mask_b)
    u = np.bitwise_or(mask_a, mask_b)
    iou = np.sum(i)/np.sum(u)

    # return the index
    return iou


def display_frame(frame, gt, zt=None, ppos=None):
    """
    Annotate the ground truth, the tracker estimate, the confidence value and the Jaccard index on the frame and display it.
    """

    # copy the frame
    annotated_frame = frame.copy()

    # if there is a bounding box, annotate some info on the frame
    if not np.isnan(gt[0]):

        # get the bounding box dimensions
        hw, hh = ((gt[2]-gt[0])//2, (gt[3]-gt[1])//2)

        # annotate the frame with the ground truth and the detection
        cv2.rectangle(annotated_frame, (gt[0], gt[1]), (gt[2], gt[3]), (255, 0, 0), 1)
        cv2.rectangle(annotated_frame, (zt.x-hw, zt.y-hh), (zt.x+hw, zt.y+hh), (0, 255, 0), 1)

        # annotate the filter result if available
        if ppos is not None:
            tl, br = ((ppos.x-hw, ppos.y-hh), (ppos.x+hw, ppos.y+hh))
            cv2.circle(annotated_frame, (ppos.x, ppos.y), 2, (0, 255, 255), -1)
            cv2.rectangle(annotated_frame, tl, br, (0, 255, 255), 1)

    # display the frame
    cv2.imshow('annotated_frame', annotated_frame)
    cv2.waitKey(1)


def plot_particles(ps):
    """
    Plots particle distribution of state info.
    """

    # plot the position
    plt.subplot(221)
    plt.scatter([p.x for p in ps[:]], [p.y for p in ps[:]], c='blue')
    plt.axis([0, 320, 0, 240])

    # plot the velocity
    plt.subplot(222)
    plt.scatter([p.vx for p in ps[:]], [p.vy for p in ps[:]], c='red')
    plt.axis([-100, 100, -100, 100])

    # show it and clear old points
    plt.show(block=False)
    plt.pause(0.001)
    plt.gcf().clear()


def sample_motion_model(stm1):
    """
    Generates a sample st_ ~ p(st_ | stm1) for the motion model estimate.
    """

    # calculates the state transition. the noise is assumed to be gaussian
    # the movement model is very simplistic, given that the camera movements
    # and object movements do not happen with constant velocity or constant acceleration
    x = stm1.x + stm1.vx + np.random.normal(0.0, 2.0)
    y = stm1.y + stm1.vy + np.random.normal(0.0, 2.0)
    vx = stm1.vx + np.random.normal(0.0, 0.5)
    vy = stm1.vy + np.random.normal(0.0, 0.5)
    
    # return estimated pose
    return State(x, y, vx, vy)


def calc_importance(st_, zt):
    """
    Calculates the importance of a particle according to the observation model and the estimate associated with the particle.
    """

    # values for the mean (u) and standard deviation (o) for p(zt | st_[m])
    # note that the distribution is assumed to be normal around the real value
    # (which is most likely not true, given that the 'sensor' is the template matcher)
    ux = st_.x
    uy = st_.y
    ox = 0.5
    oy = 0.5 # tuned

    # calculates the probability p(zt | st_[m]), assumed to be normal
    wt = 1/(2 * np.pi * ox * oy) * np.exp(-1/2 * (np.square((zt.x-ux)/(ox)) + np.square((zt.y-uy)/(oy))))

    # returns the importance
    return wt


def main():
    """
    Main function.
    """

    # setup argument parser
    ap = argparse.ArgumentParser('Settings.')

    # mode arguments
    ap.add_argument('-d', '--dataset', type=str,
        help='Dataset name.')
    ap.add_argument('-m', '--matcher', type=str, default='sqdiff',
        help='Method to be used with the template matcher. Can be \'ccoeff\', \'ccorr\' or \'sqdiff\' (default).')
    ap.add_argument('-n', '--num-particles', type=int, default=1000,
        help='Number of particles to be used in the filter. Default is 1000.')
    ap.add_argument('-p', '--particle', action='store_true',
        help='Enables particle filtering.')

    # parse the arguments
    args = vars(ap.parse_args())

    # get data from the disk
    annotations, img_list = get_ground_truth(args['dataset'])

    # initialize the restart flag, the fail count and the jaccard list
    restart = True
    failCount = 0
    jacs = []

    # initialize the template and the matcher
    template = None
    matcher = args['matcher']

    # number of particles
    M = args['num_particles']

    # particle sets
    psStm1 = [] # particle set for previous state (stm1)
    psSt_ = [] # particle set for state estimate (st_)
    psSt = [] # particle set for state (st)
    psWt = [] # particle set for importance (wt)

    # iterate through all images
    for (img_file, gt) in zip(img_list, annotations):

        # load the original frame
        frame = cv2.imread(img_file)

        # if there is a valid bbox
        if not np.isnan(gt[0]):

            # get the bbox info to int
            gt = [int(v) for v in gt]

            # restart the tracker (if the previous iteration failed or if there was no previous iteration)
            if restart:
                x0, x1 = (max(gt[0], 0), min(gt[2], frame.shape[1]))
                y0, y1 = (max(gt[1], 0), min(gt[3], frame.shape[0]))
                template = frame[y0:y1, x0:x1, :]
                if args['particle']:
                    psStm1 = [State((x0+x1)/2, (y0+y1)/2, 0, 0) for m in range(M)]
                restart = False
            
            # detect the object being tracked
            zt = find_object(frame, template, matcher)

            if args['particle']:

                # sample st_[m] ~ p(st_ | stm1[m]) for all particles
                psSt_ = [sample_motion_model(stm1) for stm1 in psStm1]

                # determine wt[m] = p(zt | st_[m]) for all particles
                psWt = [calc_importance(st_, zt) for st_ in psSt_]
                psWt = psWt/np.sum(psWt)

                # draw i with probability ~ wt[i] and add st[i] to St
                psSt = np.random.choice(psSt_, size=M, p=psWt)

                # plot the particle distribution
                # plot_particles(psSt)
            
                # calculate the jaccard index
                psx = int(np.mean([p.x for p in psSt]))
                psy = int(np.mean([p.y for p in psSt]))
                hw, hh = ((gt[2]-gt[0])//2, (gt[3]-gt[1])//2)
                jac = jaccard((psx-hw, psy-hh, psx+hw, psy+hh), gt)
                jacs.append(jac)

                # display the annotated frame
                display_frame(frame, gt, zt, State(psx, psy))

                # next iteration
                psStm1 = psSt

            else:

                # calculate the jaccard index
                hw, hh = ((gt[2]-gt[0])//2, (gt[3]-gt[1])//2)
                jac = jaccard((zt.x-hw, zt.y-hh, zt.x+hw, zt.y+hh), gt)
                jacs.append(jac)

                # display the annotated frame
                display_frame(frame, gt, zt)

            # signal a fail if there is no intersection
            if jac <= 0.0:
                restart = True
                failCount += 1

        # if the case there is no bounding box, skip the whole algorithm
        else:
            display_frame(frame, gt)
        
    # generate the metrics
    jac = np.mean(jacs)
    rs = np.exp(-30 * failCount / len(img_list))
    if args['particle']:
        tracker = 'Template Matcher + Particle Filtering'
    else:
        tracker = 'Template Matcher'
    print("Tracker = {}\t Dataset = {}".format(tracker, args["dataset"]))
    print("A robustez foi de {}\nA media do Jaccard foi {}\n".format(rs, jac))


if __name__ == "__main__":
    main()