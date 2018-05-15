# import the necessary packages
import numpy as np
import cv2
import argparse
import os
import dlib

def gen_split(req, root):
    """
    Generate image and ground-truth lists for train, test and validation sets.
    """
    # use the reduced sfa dataset structure
    if req == 1:

        # get training image id list
        img_id_list = []
        ori_paths = sorted(os.listdir(os.path.join(root, 'ORI', 'train')))
        gt_paths = sorted(os.listdir(os.path.join(root, 'GT', 'train')))
        for (img_path, gt_path) in zip(ori_paths, gt_paths):

            # convert image id to 4 digit number and store it
            img_id = img_path.split('.')[0].zfill(4)
            img_id_list.append(img_id)
        
        # sort the training images according to the image id list
        train_imgs = [os.path.join(root, 'ORI', 'train', img) for (_, img) in sorted(zip(img_id_list, ori_paths))]
        train_gts = [os.path.join(root, 'GT', 'train', gt) for (_, gt) in sorted(zip(img_id_list, gt_paths))]

        # get testing image id list
        img_id_list = []
        ori_paths = sorted(os.listdir(os.path.join(root, 'ORI', 'test')))
        gt_paths = sorted(os.listdir(os.path.join(root, 'GT', 'test')))
        for (img_path, gt_path) in zip(ori_paths, gt_paths):

            # convert image id to 4 digit number and store it
            img_id = img_path.split('.')[0].zfill(4)
            img_id_list.append(img_id)
        
        # sort the testing images according to the image id list
        test_imgs = [os.path.join(root, 'ORI', 'test', img) for (_, img) in sorted(zip(img_id_list, ori_paths))]
        test_gts = [os.path.join(root, 'GT', 'test', gt) for (_, gt) in sorted(zip(img_id_list, gt_paths))]

        # set validation lists to None
        val_imgs = None
        val_gts = None

    # use the original sfa dataset structure
    else:

        # get image id list
        img_id_list = []
        ori_paths = sorted(os.listdir(os.path.join(root, 'ORI')))
        gt_paths = sorted(os.listdir(os.path.join(root, 'GT')))
        for (img_path, gt_path) in zip(ori_paths, gt_paths):

            # convert image id to 4 digit number and store it
            img_id = img_path.split('(')[1].split(')')[0].zfill(4)
            img_id_list.append(img_id)

        # sort the images according to the image list
        set_imgs = [os.path.join(root, 'ORI', img) for (_, img) in sorted(zip(img_id_list, ori_paths))]
        set_gts = [os.path.join(root, 'GT', gt) for (_, gt) in sorted(zip(img_id_list, gt_paths))]

        # split the set into training, testing and validation
        train_imgs = set_imgs[0:782]
        train_gts = set_gts[0:782]
        val_imgs = set_imgs[782:950]
        val_gts = set_gts[782:950]
        test_imgs = set_imgs[950:1118]
        test_gts = set_gts[950:1118]

    # return the split
    return train_imgs, train_gts, val_imgs, val_gts, test_imgs, test_gts


def get_segmentation_mask(method, img):
    """
    Determine the segmentation mask according to given method (supported: ycbcr, hsv and face).
    """
    # use the YCbCr color space to segment skin
    if method == 'ycbcr':

        # convert image to ycbcr
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        
        # define the upper and lower boundaries of the YCbCr pixel
        # intensities to be considered 'skin'
        lower = np.array([50, 133, 80], dtype = "uint8")
        upper = np.array([255, 173, 120], dtype = "uint8")

        # determine the YCbCr pixel intensities that fall into
        # the specified upper and lower boundaries
        skin_mask = cv2.inRange(img_ycrcb, lower, upper)

    # use the HSV color space to segment skin
    elif method == 'hsv':

        # convert image to hsv
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

        # define the upper and lower boundaries of the HSV pixel
        # intensities to be considered 'skin'
        lower = np.array([0, 71, 80], dtype = "uint8")
        upper = np.array([40, 173, 255], dtype = "uint8")

        # determine the HSV pixel intensities that fall into
        # the specified upper and lower boundaries
        skin_mask = cv2.inRange(img_hsv, lower, upper)

    # detect the face and get skin color from the crop
    elif method == 'face':

        # detect face using dlib
        detector = dlib.get_frontal_face_detector()
        dets = detector(img, 1)

        # get face crop
        crop = img[dets[0].top():dets[0].bottom()+1, dets[0].left():dets[0].right()+1, :]
        # crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV_FULL)
        crop_ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCR_CB)

        # get average colors
        # hsv_mean = np.mean(crop_hsv.reshape(-1, crop_hsv.shape[2]), axis=0)
        # hsv_std = np.std(crop_hsv.reshape(-1, crop_hsv.shape[2]), axis=0)
        ycrcb_mean = np.mean(crop_ycrcb.reshape(-1, crop_ycrcb.shape[2]), axis=0)
        ycrcb_std = np.std(crop_ycrcb.reshape(-1, crop_ycrcb.shape[2]), axis=0)

        # check for pixels in this range
        # mask_hsv = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL),
        #     hsv_mean-1.5*hsv_std, hsv_mean+1.5*hsv_std)
        mask_ycrcb = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB),
            ycrcb_mean-1.6*ycrcb_std, ycrcb_mean+1.6*ycrcb_std)

        # pick the ycbcr mask
        skin_mask = mask_ycrcb

    # unsupported
    else:
        print('Unsupported method.')
        skin_mask = None

    # return the skin mask
    return skin_mask


# if main file
if __name__ == '__main__':

    # setup argument parser
    ap = argparse.ArgumentParser('Computes the skin color segmentation from color spaces.')

    # input image folders
    ap.add_argument('--dataset', type=str,
        help='Path to dataset root folder. Must contain ORI and GT folders.')
    ap.add_argument('--req', type=int,
        help='Specify the requirement being tested. Must be provided since the dataset folder structure and naming schemes are different for each requirement.')
    ap.add_argument('--method', type=str,
        help='Select segmentation method (ycbcr, hsv or face).')
    ap.add_argument('--verbose', action='store_true',
        help='If present, show values during computations.')

    # parse the arguments
    args = vars(ap.parse_args())

    # get image and groud-truth splits
    train_imgs, train_gts, val_imgs, val_gts, test_imgs, test_gts = gen_split(args['req'], args['dataset'])

    # for all test images, get segmentation and performance measures
    accuracy_mean = 0
    jaccard_mean = 0
    prog_counter = 0
    for img_path, gt_path in zip(test_imgs, test_gts):

        # read the image and the ground truth
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)
        
        # create mask from ground truth
        gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        _, gt_mask = cv2.threshold(gt_gray, 0, 255, cv2.THRESH_BINARY)

        # get segmentation mask using desired method
        skin_mask = get_segmentation_mask(args['method'], img)
        # cv2.imshow("prediction / ground-truth", np.hstack((skin_mask, gt_mask)))
        # cv2.waitKey(0)

        # calculate the jaccard index
        i = np.bitwise_and(skin_mask, gt_mask)
        u = np.bitwise_or(skin_mask, gt_mask)
        jaccard = np.sum(i)/np.sum(u)
        jaccard_mean += jaccard
        # print("Jaccard index: {}".format(jaccard))

        # calculate the accuracy
        xnor = np.invert(np.bitwise_xor(skin_mask, gt_mask))
        accuracy = (np.sum(xnor) / 255) / (xnor.shape[0] * xnor.shape[1])
        accuracy_mean += accuracy
        # print("Accuracy: {}".format(accuracy))

        # if working on whole set, show progress
        if args['req'] == 2:
            prog_counter += 1
            comp = int(50 * (prog_counter / len(test_imgs)))
            miss = 50 - comp
            print("Progress: [{}{}]".format('#'*comp, ' '*miss))

    # print mean accuracy
    jaccard_mean /= len(test_imgs)
    accuracy_mean /= len(test_imgs)
    print('Jaccard index: {}\tAccuracy: {}'.format(jaccard_mean, accuracy_mean))