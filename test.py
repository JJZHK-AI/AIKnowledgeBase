'''
@author: zhangkai
@license: (C) Copyright 2017-2023
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: test.py
@time: 2019-08-29 10:40:29
@desc: 
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np


def nms_python(bboxes, threshold):
    '''
    NMS: first sort the bboxes by scores ,
        keep the bbox with highest score as reference,
        iterate through all other bboxes,
        calculate Intersection Over Union (IOU) between reference bbox and other bbox
        if iou is greater than threshold,then discard the bbox and continue.

    Input:
        bboxes(numpy array of tuples) : Bounding Box Proposals in the format (x_min,y_min,x_max,y_max).
        pscores(numpy array of floats) : confidance scores for each bbox in bboxes.
        threshold(float): Overlapping threshold above which proposals will be discarded.

    Output:
        filtered_bboxes(numpy array) :selected bboxes for which IOU is less than threshold.
    '''
    # Unstacking Bounding Box Coordinates
    bboxes = bboxes.astype('float')
    x_min = bboxes[:, 0]
    y_min = bboxes[:, 1]
    x_max = bboxes[:, 2]
    y_max = bboxes[:, 3]
    psocres = bboxes[:, 4]
    # Sorting the pscores in descending order and keeping respective indices.
    sorted_idx = psocres.argsort()[::-1]
    # Calculating areas of all bboxes.Adding 1 to the side values to avoid zero area bboxes.
    bbox_areas = (x_max - x_min + 1) * (y_max - y_min + 1)

    # list to keep filtered bboxes.
    filtered = []
    while len(sorted_idx) > 0:
        # Keeping highest pscore bbox as reference.
        rbbox_i = sorted_idx[0]
        # Appending the reference bbox index to filtered list.
        filtered.append(rbbox_i)

        # Calculating (xmin,ymin,xmax,ymax) coordinates of all bboxes w.r.t to reference bbox
        overlap_xmins = np.maximum(x_min[rbbox_i], x_min[sorted_idx[1:]])
        overlap_ymins = np.maximum(y_min[rbbox_i], y_min[sorted_idx[1:]])
        overlap_xmaxs = np.minimum(x_max[rbbox_i], x_max[sorted_idx[1:]])
        overlap_ymaxs = np.minimum(y_max[rbbox_i], y_max[sorted_idx[1:]])

        # Calculating overlap bbox widths,heights and there by areas.
        overlap_widths = np.maximum(0, (overlap_xmaxs - overlap_xmins + 1))
        overlap_heights = np.maximum(0, (overlap_ymaxs - overlap_ymins + 1))
        overlap_areas = overlap_widths * overlap_heights

        # Calculating IOUs for all bboxes except reference bbox
        ious = overlap_areas / (bbox_areas[rbbox_i] + bbox_areas[sorted_idx[1:]] - overlap_areas)

        # select indices for which IOU is greather than threshold
        delete_idx = np.where(ious > threshold)[0] + 1
        delete_idx = np.concatenate(([0], delete_idx))

        # delete the above indices
        sorted_idx = np.delete(sorted_idx, delete_idx)

    # Return filtered bboxes
    return bboxes[filtered].astype('int')


if __name__ == '__main__':
    COLOR = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
        [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
        [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128]
    ]
    bboxes = np.array([
        (165, 127, 296, 455, 0.8),
        (148, 142, 257, 459, 0.95),
        (142, 137, 270, 465, 0.81),
        (129, 122, 302, 471, 0.85),
        (327, 262, 604, 465, 0.94),
        (349, 253, 618, 456, 0.83),
        (369, 248, 601, 470, 0.82)
    ])
    sampleimage = cv2.imread("Images/03/02/03_02_004.jpg")
    sampleimage = cv2.cvtColor(sampleimage,cv2.COLOR_BGR2RGB)
    sampleimageallbb = sampleimage.copy()
    sampleimagenmsbb = sampleimage.copy()
    # Sample BBoxes and corresponding scores.

    # Drawing all rectangular bboxes on original image
    for index, bbox in enumerate(bboxes):
        top_left = int(bbox[0]), int(bbox[1])
        bottom_right = int(bbox[2]), int(bbox[3])
        cv2.rectangle(sampleimageallbb, top_left, bottom_right, COLOR[index], 2)
    # Getting nms filtered bboxes
    bboxes_after_nms = nms_python(bboxes, 0.3)

    # Drawing nms filtered rectangular bboxes on original image
    for bbox in bboxes_after_nms:
        top_left = bbox[0], bbox[1]
        bottom_right = bbox[2], bbox[3]
        cv2.rectangle(sampleimagenmsbb, top_left, bottom_right, (255, 0, 0), 2)

    image_list = [sampleimageallbb, sampleimagenmsbb]
    titles = ["BBoxes before NMS", "BBoxes after NMS"]

    fig, axes = plt.subplots(1, 2, figsize=(30,20))
    for axis, (image, title) in zip(axes, zip(image_list, titles)):
        axis.imshow(image)
        axis.axis('off')
        axis.set_title(title)
    plt.show()