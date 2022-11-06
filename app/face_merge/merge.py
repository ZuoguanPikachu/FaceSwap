import numpy as np
import cv2


def mer_face(imgpath1, imgpath2, getpoints_func, erode_size=None, erode_iterations=None, blur_size=None):
    img1 = cv2.imread(imgpath1)
    img2 = cv2.imread(imgpath2)

    points = getpoints_func(img1)
    
    src_mask = np.zeros(img2.shape, img2.dtype)
    poly = np.array(points, np.int32)
    src_mask = cv2.fillPoly(src_mask, [poly], (255, 255, 255))

    if (erode_size is not None) and (erode_iterations is not None): 
        src_mask = cv2.erode(src_mask, np.ones((erode_size, erode_size), np.uint8), iterations=erode_iterations)
    
    if blur_size is not None:
        src_mask = cv2.GaussianBlur(src_mask, (blur_size, blur_size), 0, 0)

    bounding_rect = cv2.boundingRect(cv2.cvtColor(src_mask, cv2.COLOR_RGB2GRAY))
    center = (int(bounding_rect[0]+bounding_rect[2]/2), int(bounding_rect[1]+bounding_rect[3]/2))

    merface = cv2.seamlessClone(img2, img1, src_mask, center, cv2.NORMAL_CLONE)

    return merface
