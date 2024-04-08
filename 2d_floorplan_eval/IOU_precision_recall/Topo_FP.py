#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
from random import randrange
import numpy as np
import sys
from scipy.optimize import linear_sum_assignment

from .Utility_FP import Utility_FP

class Topo_FP(object):
    
    @staticmethod
    def compute_betti_error_patch(gth, pred, connectivity, patch_size, N):
        '''
        Random sample patches and compute betti number from them, returns the mean and std error
        @gth: ground truth floor plan
        @pred: prediction floor plan, should have the same shape with gth
        @connectivity: connectivity for computing betti error
        @patch_size: the size of the sample patch
        @N: integer, number of samples
        '''
        gth = cv2.cvtColor(gth, cv2.COLOR_BGR2GRAY)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        gth[gth != 255] = 0
        pred[pred != 255] = 0
        
        # Random patch sampling
        shape = gth.shape
        discrepancy = [0] * N
        valid_num = 0
        for i in range(N):
            x_ = randrange(shape[1] - patch_size + 1)
            y_ = randrange(shape[0] - patch_size + 1)
            patch_gth  = Utility_FP.extract_patch_topleft(x_, y_, patch_size, gth)
            patch_pred = Utility_FP.extract_patch_topleft(x_, y_, patch_size, pred)
            if patch_gth is not None:
                valid_num = valid_num + 1
                cnt_gth, hry_gth, red_gth = Utility_FP.compute_bnd_red_cv(patch_gth, 0, 255, connectivity)
                cnt_prd, hry_prd, red_prd = Utility_FP.compute_bnd_red_cv(patch_pred, 0, 255, connectivity)
                discrepancy[i] = np.abs(red_gth[0] - red_prd[0])
        return np.sum(discrepancy) / valid_num

    @staticmethod
    def compute_room_matching(img1, img2, units, area_threshold):
        '''
        @img1: 3 channel image of groudth truth floorplan
        @img2: 3 chaneel image of user floorplan
        @units: length of each pixel, has to be "20cm"
        @area_threshold: the threshold to determine a room
        '''
        if units != "20cm":
            print("Invalid pixel length, has to be 20cm")
            sys.exit()
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img1[img1 != 255] = 0
        img2[img2 != 255] = 0
        cnt1, hry1, red1 = Utility_FP.compute_bnd_red_cv(img1, 0, 255, 8)
        cnt2, hry2, red2 = Utility_FP.compute_bnd_red_cv(img2, 0, 255, 8)

        ind1 = []
        ind2 = []
        num1 = 0
        for i in range(2, red1[0]):
            if np.sum(red1[1] == i) >= area_threshold:
                num1 = num1 + 1
                ind1.append(i)
        num2 = 0
        for i in range(2, red2[0]):
            if np.sum(red2[1] == i) >= area_threshold:
                num2 = num2 + 1
                ind2.append(i)
        if num1 == 0 or num2 == 0:
            return 0.0

        cost_matrix = np.ones((num1, num2), dtype=np.float32) * sys.float_info.max
        num1 = 0
        for i in range(2, red1[0]):
            if np.sum(red1[1] == i) >= area_threshold:
                a1 = (red1[1] == i) * 1
                num2 = 0
                for j in range(2, red2[0]):
                    if np.sum(red2[1] == j) >= area_threshold:
                        a2 = (red2[1] == j) * 1
                        intersection = np.sum(np.multiply(a1, a2))
                        union = a1 + a2
                        union[union > 1] = 1
                        union = np.sum(union)
                        score = intersection / union
                        if score > 0:
                            cost_matrix[num1][num2] = 1/score
                        num2 = num2 + 1
                num1 = num1 + 1
  
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_intersection = 0
        total_union = 0
        for i in range(len(row_ind)):
            ind1_ = row_ind[i]
            ind2_ = col_ind[i]
            a1_ = (red1[1] == ind1[ind1_]) * 1
            a2_ = (red2[1] == ind2[ind2_]) * 1
            intersection = np.sum(np.multiply(a1_, a2_))
            union = a1_ + a2_
            union[union > 1] = 1
            union = np.sum(union)
            total_intersection = total_intersection + intersection
            total_union = total_union + union
        return total_intersection / total_union