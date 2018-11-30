import getscore
import glob
import cv2

test_img_path = 'RUNS/crackSeg_2018_11_26_21.52/test_images/'
gt_img_path = 'DATA/data_crack_correct/training/'

test_imgs_names = sorted(glob.glob(test_img_path + '*.png'))
num = len(test_imgs_names)
gt_imgs_names = sorted(glob.glob(gt_img_path + '*_gt.png'))[-num:]
#print(test_imgs_names)
#print(gt_imgs_names)
for img_name, gt_name in zip(test_imgs_names, gt_imgs_names):
    img = cv2.imread(img_name, 0)
    gt = cv2.imread(gt_name, 0)
    scores = getscore.get_score(img, gt)
    print(scores[:-1])
