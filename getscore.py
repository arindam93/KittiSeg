import base64
import numpy as np
from io import BytesIO
from matplotlib import colors, image, pyplot
from PIL import Image
from scipy import ndimage

MIN_DISTANCE = 5
MAX_DISTANCE = 10
MAX_PENALTY = MAX_DISTANCE - MIN_DISTANCE

def get_score(img, truth_img):
    return s_hausdorff_distance(img, truth_img)

"""
img_a np.array
img_b np.array
"""
def s_hausdorff_distance(a_img, b_img):
    a_dist = get_image_dist(a_img, MIN_DISTANCE, MAX_PENALTY)
    a_area = float(np.count_nonzero(a_img))
    b_dist = get_image_dist(b_img, MIN_DISTANCE, MAX_PENALTY)
    b_area = float(np.count_nonzero(b_img))

    m, n = a_img.shape

    # create color image
    color_img = create_color_image(a_img, b_img, a_dist, b_dist)
    color_img_base64 = convert_to_base64(color_img)
    # color_img_base64 = 'TODO'

    # calculate fp, fn, and score
    if a_area > 0:
        # TODO check if correct
        fp_error = float(np.sum(np.multiply(a_img, b_dist), axis = None)) / a_area
    else:
        fp_error = 0

    if b_area > 0:
        fn_error = float(np.sum(np.multiply(b_img, a_dist), axis = None)) / b_area
    else:
        fn_error = 0

    if a_area == 0:
        fn_error = (b_area * MAX_PENALTY) / (m * n / 100)
        if fn_error > MAX_PENALTY:
            fn_error = MAX_PENALTY

    if b_area == 0:
        fp_error = (a_area * MAX_PENALTY) / (m * n / 100)
        if fp_error > MAX_PENALTY:
            fp_error = MAX_PENALTY

    shd = max(fp_error, fn_error)
    # remove rounding errors
    shd = min(shd, MAX_PENALTY)
    score = 100 - ((shd * 100) / MAX_PENALTY)
    fp = fp_error / MAX_PENALTY
    fn = fn_error / MAX_PENALTY
    return (score, fp, fn, color_img_base64)

"""
img np.array
min_distance int
max_penalty int
"""
def get_image_dist(img, min_distance, max_penalty):
    img[img > 1] = 1
    img_dist = ndimage.distance_transform_edt(1 - img)
    img_dist = img_dist - min_distance
    img_dist[img_dist < 0] = 0
    img_dist[img_dist > max_penalty] = max_penalty
    return img_dist

def create_color_image(a_img, b_img, a_dist, b_dist):
    layer1 = np.multiply(b_img, a_dist) / MAX_PENALTY
    layer1[b_img == 0] = -1
    layer1 = create_color_map(layer1, 0.5, 0.7)

    layer2 = np.multiply(a_img, b_dist) / MAX_PENALTY
    layer2[a_img == 0] = -1
    layer2 = create_color_map(layer2, 0.8333, 1)

    return combine_color_maps(layer1, layer2)

def create_color_map(gray_img, hue, value):
    m, n = gray_img.shape
    color_img = np.zeros((m, n, 3))
    color_img[:, :, 0] = hue
    value_pane = np.zeros((m, n))
    value_pane[gray_img > -1] = value
    color_img[:, :, 2] = value_pane
    gray_img[gray_img < 0] = 0
    color_img[:, :, 1] = gray_img
    return colors.hsv_to_rgb(color_img)

def combine_color_maps(layer1, layer2):
    combined_map = layer1 + layer2
    combined_map[combined_map > 1] = 1
    return combined_map

def convert_to_base64(img_array):
    buffer = BytesIO()
    pyplot.imsave(buffer, img_array, format='PNG')
    pyplot.close()
    return base64.b64encode(buffer.getvalue()).decode('utf-8')