#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import struct
import json
import numpy as np

class FileIO_FP(object):
    
    @staticmethod
    def wrt_geometry_2_JSON(geometry, out_path, in_inches):
        '''
        @geometry: output from extract_geometry_fromGDB with flag "obj_output" set as True
        @out_path: the path to the output .json file
        @in_inches: bool, if true output in inches, otherwise in meters
        '''
        num_layers = len(geometry)
        num_struct = [0] * num_layers
        for i in range(num_layers):
            num_struct[i] = len(geometry[i])

        data = {}
        data['header'] = {'layer number': num_layers, 'structure number': num_struct}
        for i in range(num_layers):
            layer_number = "layer " + str(i)
            data[layer_number] = {'layer name': geometry[i][0][0], 'points': []}
            for j in range(num_struct[i]):
                if in_inches:
                    coords_in_inches = np.asarray(geometry[i][j][2:])/12*0.3048
                    data[layer_number]['points'].append({'point number': geometry[i][j][1], 'coordinates': list(coords_in_inches)}) # output in inches
                else:
                    data[layer_number]['points'].append({'point number': geometry[i][j][1], 'coordinates': geometry[i][j][2:]}) # output in meters
        with open(out_path, 'w') as fout:
            json.dump(data, fout, indent=2)
            
    @staticmethod
    def read_geometry_JSON(in_path, cvt2_units):
        '''
        @in_path: the path to the input .obj file. format of .obj file has to follow the specifications
        @cvt2_units: string, can be "1inch", "20cm", "10cm", "1cm"
        '''
        with open(in_path) as fin:
            data = json.load(fin)
            num_layers = data['header']['layer number']
            geometry_result = [None] * num_layers
            
            if num_layers > 0:
                num_struct = data['header']['structure number']
                for i in range(num_layers):
                    tmp_ = [None] * num_struct[i]
                    geometry_result[i] = tmp_

            for i in range(num_layers):
                layer_number = "layer " + str(i)
                layer_name = data[layer_number]['layer name']
                for j in range(num_struct[i]):
                    pt_num = data[layer_number]['points'][j]['point number']
                    pt_coords = data[layer_number]['points'][j]['coordinates']
                    if cvt2_units == "1inch":
                        pt_coords = np.asarray(pt_coords)/0.3048*12
                        pt_coords = pt_coords.tolist()
                    elif cvt2_units == "1cm":
                        pt_coords = np.asarray(pt_coords)*100
                        pt_coords = pt_coords.tolist()
                    elif cvt2_units == "10cm":
                        pt_coords = np.asarray(pt_coords)*10
                        pt_coords = pt_coords.tolist()
                    elif cvt2_units == "20cm":
                        pt_coords = np.asarray(pt_coords)*5
                        pt_coords = pt_coords.tolist()
                    else:
                        print("Invalid units request...")
                        sys.exit()
                    row_ = [layer_name, pt_num] + pt_coords
                    geometry_result[i][j] = row_
            return geometry_result

