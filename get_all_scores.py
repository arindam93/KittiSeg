import getscore
import os
import glob
import cv2
import numpy as np

test_img_path = 'RUNS/crackSeg_2018_12_04_16.02/test_images/'
#test_img_path = 'RUNS/crackSeg_2018_11_26_21.52/test_images/'
gt_img_path = 'DATA/data_crack_correct/training/'

#num = len(test_imgs_names)
num = 56
test_imgs_names = sorted(glob.glob(test_img_path + '*.png'))[:num]
gt_imgs_names = sorted(glob.glob(gt_img_path + '*_gt.png'))[:num]
#print(test_imgs_names)
#print(gt_imgs_names)
avg = 0
avg_pixel_diff_pct = 0
thresh = 60

binary_output_path = os.path.join(test_img_path, 'binary_output')
if not os.path.exists(binary_output_path):
    os.makedirs(binary_output_path)

for i, (img_name, gt_name) in enumerate(zip(test_imgs_names, gt_imgs_names)):
    if i % 4 != 3:
        continue
    img = cv2.imread(img_name, 0)
    img[img <= thresh] = 0
    img[img > thresh] = 1
    
    basename = os.path.basename(img_name)
    binary_img_path = os.path.join(binary_output_path, basename)
    cv2.imwrite(binary_img_path, img * 255)
    gt = cv2.imread(gt_name, 0)
    gt[gt > 1] = 1

    num_pixels = img.shape[0] * img.shape[1]
    tmp = (img == gt)
    #print(tmp.shape)
    #print(np.max(img), np.max(gt))
    #print(np.max(tmp), np.min(tmp))
    #pixel_diff = np.sum(np.abs(img - gt))
    #if pixel_diff > num_pixels:
    #    print(pixel_diff, num_pixels)
    pixel_diff_pct = np.sum(tmp) / float(num_pixels)
    avg_pixel_diff_pct += pixel_diff_pct
    scores = getscore.get_score(img, gt)
    avg += scores[0]
    print(os.path.basename(img_name), pixel_diff_pct, scores[:-1])

print('avg score: {}'.format(avg / float(num / 4)))
print('avg pixel diff pct: {}'.format(avg_pixel_diff_pct / float(num / 4)))
