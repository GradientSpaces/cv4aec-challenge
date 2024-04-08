#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from .Topo_FP import Topo_FP
from .FileIO_FP import FileIO_FP
from .Viewer_FP import Viewer_FP
from .Utility_FP import Utility_FP
from .Conversion_DWG_FP import Conversion_DWG_FP

import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_precision_recall_helper(path1, path2, units, threshold):
    '''
    @path1: path to ground truth jason file
    @path2: path to generated jason file
    @units: length of each pixel, has to be "1cm"
    @threshold: threshold to match the points, in cm, 
     pass in 5, 10, 20 as thresholds corresponding to 5, 10, 20 cm
    '''
    geo1 = FileIO_FP.read_geometry_JSON(path1, "1cm")
    geo2 = FileIO_FP.read_geometry_JSON(path2, "1cm")
    x1, y1 = Conversion_DWG_FP.extract_all_points(geo1)
    x2, y2 = Conversion_DWG_FP.extract_all_points(geo2)
    cost_matrix = Utility_FP.pairwise_distance(x1, y1, x2, y2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_pts = 0
    for i in range(len(row_ind)):
        if cost_matrix[row_ind[i]][col_ind[i]] < threshold * threshold:
            matched_pts = matched_pts + 1
    precision = matched_pts / len(x2)
    recall = matched_pts / len(x1)
    return precision, recall

def compute_precision_recall(path1, path2, units):
    '''
    @path1: path to ground truth jason file
    @path2: path to generated jason file
    @units: length of each pixel, has to be "1cm"
    '''
    if units != "1cm":
        print("Invalid pixel length, has to be 1cm")
        sys.exit()
    p1, r1 = compute_precision_recall_helper(path1, path2, units, 5)
    p2, r2 = compute_precision_recall_helper(path1, path2, units, 10)
    p3, r3 = compute_precision_recall_helper(path1, path2, units, 20)
    return [p1, p2, p3], [r1, r2, r3]

def compute_room_IOU(path1, path2, units, area_threshold):
    '''
    @path1: path to ground truth jason file
    @path2: path to generated jason file
    @units: length of each pixel
    @area_threshold: the threshold to determine a room
    '''
    geo1 = FileIO_FP.read_geometry_JSON(path1, units)
    geo2 = FileIO_FP.read_geometry_JSON(path2, units)
    geo1 = Conversion_DWG_FP.cvt_geometry_format_obj2drw(geo1)
    geo2 = Conversion_DWG_FP.cvt_geometry_format_obj2drw(geo2)
    x1_1, y1_1, x2_1, y2_1 = Utility_FP.cvt_geometry2list(geo1)
    x1_2, y1_2, x2_2, y2_2 = Utility_FP.cvt_geometry2list(geo2)
    shape = Viewer_FP.determine_curtain_size_sync(x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2)
    img1  = Viewer_FP.plot_layers(x1_1, y1_1, x2_1, y2_1, [-1], shape, 1)
    img2  = Viewer_FP.plot_layers(x1_2, y1_2, x2_2, y2_2, [-1], shape, 1)
    iou = Topo_FP.compute_room_matching(img1, img2, units, area_threshold)
    return iou

def compute_Betti_error(path1, path2, patch_size, N):
    '''
    @path1: path to ground truth jason file
    @path2: path to generated jason file
    @patch_size: integer, size of the sample patch
    @N: integer, number of samples
    '''
    geo1 = FileIO_FP.read_geometry_JSON(path1, "20cm")
    geo2 = FileIO_FP.read_geometry_JSON(path2, "20cm")
    geo1 = Conversion_DWG_FP.cvt_geometry_format_obj2drw(geo1)
    geo2 = Conversion_DWG_FP.cvt_geometry_format_obj2drw(geo2)
    x1_1, y1_1, x2_1, y2_1 = Utility_FP.cvt_geometry2list(geo1)
    x1_2, y1_2, x2_2, y2_2 = Utility_FP.cvt_geometry2list(geo2)
    shape = Viewer_FP.determine_curtain_size_sync(x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2)
    img1 = Viewer_FP.plot_layers(x1_1, y1_1, x2_1, y2_1, [-1], shape, 1)
    img2 = Viewer_FP.plot_layers(x1_2, y1_2, x2_2, y2_2, [-1], shape, 1)
    betti_error = Topo_FP.compute_betti_error_patch(img1, img2, 8, patch_size, N)
    return betti_error

def compute_all(gt, user, recall_thresh="1cm", iou_tresh="20cm"):
    precision, recall = compute_precision_recall(gt, user, recall_thresh)
    iou = compute_room_IOU(gt, user, iou_tresh, 100)
    betti_error = compute_Betti_error(gt, user, 25, 500)

    return {
        'precision': precision,
        'recall': recall,
        'iou': iou,
        'betti_error': betti_error,
    }

def main():
    results = compute_all(sys.argv[1], sys.argv[2])
    print(results)

if __name__ == "__main__":
    quit(main())
