#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randrange

class Viewer_FP(object):
    
    @staticmethod
    def plot_layers(x1, y1, x2, y2, index, shape, thickness):
        '''
        @x1, y1, x2, y2: x y coordinates for two endpoints
        @index: list of integers indicating the layers to draw. If [-1] is given, draw all layers
        @shape: pre-defined shape in format of [height, width]
        @thickness: thickness of the drawn line
        '''
        img = np.ones((shape[0], shape[1]), np.uint8) * 255
        img = cv2.merge((img, img, img))
        if len(index) == 1 and index[0] == -1:
            index = np.arange(len(x1))
        if len(x1) == 0 or len(index) == 0:
            return img
        assert(len(x1) >= 1)
        assert(len(index) >= 1)
        assert(np.amax(index) < len(x1))
        colors = [None] * len(index)
        colors[0] = (0,0,0)
        for i in range(1, len(index)):
            colors[i] = (randrange(256), randrange(256), randrange(256))
        for i in range(0, len(index)):
            img = Viewer_FP.draw_coord(x1, y1, x2, y2, index[i], thickness, colors[i], img)
        return img
    
    @staticmethod
    def draw_coord(x1, y1, x2, y2, index, thickness, color, img):
        '''
        Draw the coordinates output from cvt_geometry2list on an image
        @x1: x coordinate of the left endpoint
        @y1: y coordinate of the left endpoint
        @x2: x coordinate of the right endpoint
        @y2: y coordinate of the right endpoint
        @index: the layer to draw
        @thickness: line thickness
        @color: line color
        @img: the curtain to draw the coordinates
        '''
        if index >= len(x1):
            print("Requested index exceeds the size of the layers")
        rows = img.shape[0]
        cols = img.shape[1]
        for i in range(len(x1[index])):
            x1_ = int(np.rint(x1[index][i]))
            y1_ = int(np.rint(y1[index][i]))
            x2_ = int(np.rint(x2[index][i]))
            y2_ = int(np.rint(y2[index][i]))
            if (x1_<0 or x1_>=cols or x2_<0 or x2_>=cols or y1_<0 or y1_>=rows or y2_<0 or y2_>=rows):
                print("Invalid coordinates")
            img = cv2.line(img, (x1_, y1_), (x2_, y2_), color, thickness)
        return img
    
    @staticmethod
    def determine_curtain_size_sync(x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2):
        '''
        Determine the size of the curtain
        @x1: x coordinate of the left endpoint
        @y1: y coordinate of the left endpoint
        @x2: x coordinate of the right endpoint
        @y2: y coordinate of the right endpoint
        '''
        print("Determining the size of the image...")
        x_1_min = [0.0] * len(x1_1) * 2
        y_1_min = [0.0] * len(x1_1) * 2
        x_1_max = [0.0] * len(x1_1) * 2
        y_1_max = [0.0] * len(x1_1) * 2
        x_2_min = [0.0] * len(x1_2) * 2
        y_2_min = [0.0] * len(x1_2) * 2
        x_2_max = [0.0] * len(x1_2) * 2
        y_2_max = [0.0] * len(x1_2) * 2
        for i in range(len(x1_1)):
            if (len(x1_1[i]) > 0):
                x_1_min[i*2+0] = np.amin(x1_1[i])
                x_1_min[i*2+1] = np.amin(x2_1[i])
                y_1_min[i*2+0] = np.amin(y1_1[i])
                y_1_min[i*2+1] = np.amin(y2_1[i])
            else:
                x_1_min[i*2+0] = sys.float_info.max
                x_1_min[i*2+1] = sys.float_info.max
                y_1_min[i*2+0] = sys.float_info.max
                y_1_min[i*2+1] = sys.float_info.max 
        for i in range(len(x1_2)):
            if (len(x1_2[i]) > 0):
                x_2_min[i*2+0] = np.amin(x1_2[i])
                x_2_min[i*2+1] = np.amin(x2_2[i])
                y_2_min[i*2+0] = np.amin(y1_2[i])
                y_2_min[i*2+1] = np.amin(y2_2[i])
            else:
                x_2_min[i*2+0] = sys.float_info.max
                x_2_min[i*2+1] = sys.float_info.max
                y_2_min[i*2+0] = sys.float_info.max
                y_2_min[i*2+1] = sys.float_info.max
        if len(x1_1) > 0:
            x_1_min_v = np.amin(x_1_min)
            y_1_min_v = np.amin(y_1_min)
        else:
            x_1_min_v = sys.float_info.max
            y_1_min_v = sys.float_info.max
        if len(x1_2) > 0:
            x_2_min_v = np.amin(x_2_min)
            y_2_min_v = np.amin(y_2_min)
        else:
            x_2_min_v = sys.float_info.max
            y_2_min_v = sys.float_info.max
        x_shift = 50 - min(x_1_min_v, x_2_min_v)
        y_shift = 50 - min(y_1_min_v, y_2_min_v)
        
        for i in range(len(x1_1)):
            if (len(x1_1[i]) > 0):
                x1_1[i] = x1_1[i] + x_shift
                x2_1[i] = x2_1[i] + x_shift
                y1_1[i] = y1_1[i] + y_shift
                y2_1[i] = y2_1[i] + y_shift
        for i in range(len(x1_2)):
            if (len(x1_2[i]) > 0):
                x1_2[i] = x1_2[i] + x_shift
                x2_2[i] = x2_2[i] + x_shift
                y1_2[i] = y1_2[i] + y_shift
                y2_2[i] = y2_2[i] + y_shift
        for i in range(len(x1_1)):
            if (len(x1_1[i]) > 0):
                x_1_max[i*2+0] = np.amax(x1_1[i])
                x_1_max[i*2+1] = np.amax(x2_1[i])
                y_1_max[i*2+0] = np.amax(y1_1[i])
                y_1_max[i*2+1] = np.amax(y2_1[i])
            else:
                x_1_max[i*2+0] = sys.float_info.min
                x_1_max[i*2+1] = sys.float_info.min
                y_1_max[i*2+0] = sys.float_info.min
                y_1_max[i*2+1] = sys.float_info.min
        for i in range(len(x1_2)):
            if (len(x1_2[i]) > 0):
                x_2_max[i*2+0] = np.amax(x1_2[i])
                x_2_max[i*2+1] = np.amax(x2_2[i])
                y_2_max[i*2+0] = np.amax(y1_2[i])
                y_2_max[i*2+1] = np.amax(y2_2[i])
            else:
                x_2_max[i*2+0] = sys.float_info.min
                x_2_max[i*2+1] = sys.float_info.min
                y_2_max[i*2+0] = sys.float_info.min
                y_2_max[i*2+1] = sys.float_info.min
        if len(x1_1) > 0:
            x_1_max_v = np.amax(x_1_max)
            y_1_max_v = np.amax(y_1_max)
        else:
            x_1_max_v = sys.float_info.min
            y_1_max_v = sys.float_info.min
        if len(x1_2) > 0:
            x_2_max_v = np.amax(x_2_max)
            y_2_max_v = np.amax(y_2_max)
        else:
            x_2_max_v = sys.float_info.min
            y_2_max_v = sys.float_info.min
            
        return [int(np.ceil(max(y_1_max_v, y_2_max_v)))+50, int(np.ceil(max(x_1_max_v, x_2_max_v)))+50]

