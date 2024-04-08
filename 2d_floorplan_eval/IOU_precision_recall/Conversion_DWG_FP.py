#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Conversion_DWG_FP(object):
    
    @staticmethod
    def cvt_geometry_format_obj2drw(geometry):
        '''
        geometry format in obj file: [layer_name, point_num, pt1.X, pt1.Y, ... , ptN.X, ptN.y]
        geometry format for drawing: [layer_name, endpoint1.X, endpoint1.Y, endpoint2.X, endpoint2.Y]
        This function converts from former to latter for obj drawing.
        @geometry: result from FileIO_FP.read_geometry_OBJ or from Conversion_DWG_FP.extract_geometry_fromGDB
          with flag obj_output set as True
        '''
        layer_num = len(geometry)
        geometry_result = [None] * layer_num
        for i in range(layer_num):
            tmp_ = []
            geometry_result[i] = tmp_

        for i in range(layer_num):
            struct_num = len(geometry[i])
            for j in range(struct_num):
                layer_name = geometry[i][j][0]
                point_num  = geometry[i][j][1]
                for k in range(1, point_num):
                    row_ = [layer_name, geometry[i][j][k*2], geometry[i][j][k*2+1],
                            geometry[i][j][(k+1)*2], geometry[i][j][(k+1)*2+1]]
                    geometry_result[i].append(row_)
        return geometry_result
    
    @staticmethod
    def extract_all_points(geometry):
        '''
        This function extracts all points from geometry for hungarian matching
        @geometry: result from FileIO_FP.read_geometry_OBJ or from Conversion_DWG_FP.extract_geometry_fromGDB
          with flag obj_output set as True
        '''
        x_coord = []
        y_coord = []
        layer_num = len(geometry)
        for i in range(layer_num):
            struct_num = len(geometry[i])
            for j in range(struct_num):
                point_num = geometry[i][j][1]
                for k in range(1, point_num+1):
                    x_coord.append(geometry[i][j][k*2])
                    y_coord.append(geometry[i][j][k*2+1])
        return x_coord, y_coord

