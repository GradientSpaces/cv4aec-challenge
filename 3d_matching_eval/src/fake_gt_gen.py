# This script is used to randomly modify the ground-truth JSON files in order
# to generate fake data for testing the evaluation script.
#
# To use this script, first gather the JSON files into "gt_root", see variable
# below. Then change "floor_names" to the floor names of the JSON files, not
# including the object name. Then run this script and the outputs will be in
# "fake_gt_root".
#
# You can change the random parameters below.

import os
import json

import numpy as np

from copy import deepcopy
from os.path import join as path_join


# names and directories
floor_names = ['01_OfficeLab01_Allfloors',
               '05_MedOffice_01_F2',
               '07_MedOffice_03_F3',
               '07_MedOffice_03_F5',
               '08_ShortOffice_01_F1',
               '11_MedOffice_05_F4',
               '19_MedOffice_07_F4']

gt_root = '../data/'
fake_gt_root = '../augmented/'


# random parameters
translate_prob = 0.1  # probability of applying translation
translate_x_max = 2.5  # translate each corner of wall, unit in meters
translate_y_max = 2.5
translate_z_max = 2.5

scale_prob = 0.5  # probability of applying scaling
scale_w_max = 1.2  # max scale width in terms of percentage of original
scale_h_max = 1.2  # height
scale_d_max = 1.2  # depth

rotate_prob = 0.1  # only for doors and columns
rotate_max = 5  # max rotation in degrees

add_prob = 0.1  # randomly add another wall
del_prob = 0.1  # randomly delete an existing wall
merge_prob = 0.1  # randomly merge almost collinear walls


def coin_flip(p_true):
  return np.random.choice([False, True], p=[1-p_true, p_true])


# NOTE we translate z together, to keep wall level
def random_translate_wall(wall):
  translate_x0 = np.random.uniform(-translate_x_max, translate_x_max)
  translate_x1 = np.random.uniform(-translate_x_max, translate_x_max)
  translate_y0 = np.random.uniform(-translate_y_max, translate_z_max)
  translate_y1 = np.random.uniform(-translate_y_max, translate_z_max)
  translate_z = np.random.uniform(-translate_y_max, translate_z_max)

  translate_start = np.array([translate_x0, translate_y0, translate_z])
  translate_end = np.array([translate_x1, translate_y1, translate_z])

  new_start_pt = np.array(wall['start_pt']) + translate_start
  new_end_pt = np.array(wall['end_pt']) + translate_end

  wall['start_pt'] = new_start_pt.tolist()
  wall['end_pt'] = new_end_pt.tolist()

  return wall


# NOTE there is technically no depth to walls, as that is the length
def random_scale_wall(wall):
  scale_w = np.random.uniform(0, scale_w_max)
  scale_h = np.random.uniform(0, scale_h_max)

  wall['width'] *= scale_w
  wall['height'] *= scale_h

  return wall


def get_fake_walls(real_walls):
  fake_walls = []

  for wall in real_walls:
    fake_wall = deepcopy(wall)

    # random translate
    if coin_flip(translate_prob):
      fake_wall = random_translate_wall(fake_wall)

    # random scale
    if coin_flip(scale_prob):
      fake_wall = random_scale_wall(fake_wall)

    # skip this wall, thereby deleting it
    if coin_flip(del_prob):
      continue

    # add another wall that is randomly adjusted
    if coin_flip(add_prob):
      fake_wall_2 = deepcopy(wall)
      fake_wall_2 = random_translate_wall(fake_wall_2)
      fake_wall_2 = random_scale_wall(fake_wall_2)
      fake_walls.append(fake_wall_2)

    fake_walls.append(fake_wall)

  return fake_walls


def random_translate_family(family):
  translate_x = np.random.uniform(-translate_x_max, translate_x_max)
  translate_y = np.random.uniform(-translate_y_max, translate_z_max)
  translate_z = np.random.uniform(-translate_y_max, translate_z_max)

  translate = np.array([translate_x, translate_y, translate_z])

  new_loc = np.array(family['loc']) + translate
  family['loc'] = new_loc.tolist()

  return family


def random_scale_family(family):
  scale_w = np.random.uniform(0, scale_w_max)
  scale_d = np.random.uniform(0, scale_d_max)
  scale_h = np.random.uniform(0, scale_h_max)

  family['width'] *= scale_w
  family['depth'] *= scale_d
  family['height'] *= scale_h

  return family


def random_rotate_family(family):
  rotate = np.random.uniform(-rotate_max, rotate_max)

  family['rotation'] += rotate

  return family


def get_fake_families(real_families):
  fake_families = []

  for family in real_families:
    fake_family = deepcopy(family)

    # random translate
    if coin_flip(translate_prob):
      fake_family = random_translate_family(fake_family)

    # random scale
    if coin_flip(scale_prob):
      fake_family = random_scale_family(fake_family)

    # random rotate
    if coin_flip(rotate_prob):
      fake_family = random_rotate_family(fake_family)

    # skip this family, thereby deleting it
    if coin_flip(del_prob):
      continue

    # add another family that is randomly adjusted
    if coin_flip(add_prob):
      fake_family_2 = deepcopy(family)
      fake_family_2 = random_translate_family(fake_family_2)
      fake_family_2 = random_scale_family(fake_family_2)
      fake_families.append(fake_family_2)

    fake_families.append(fake_family)

  return fake_families


if __name__ == '__main__':
    if not os.path.exists(fake_gt_root):
      os.makedirs(fake_gt_root)

    for floor_name in floor_names:
      print(floor_name)

      with open(path_join(gt_root, '%s_walls.json' % floor_name), 'r') as f:
        wall_json = json.load(f)

      with open(path_join(gt_root, '%s_doors.json' % floor_name), 'r') as f:
        door_json = json.load(f)

      with open(path_join(gt_root, '%s_columns.json' % floor_name), 'r') as f:
        column_json = json.load(f)

      fake_wall_json = get_fake_walls(wall_json)
      fake_door_json = get_fake_families(door_json)
      fake_column_json = get_fake_families(column_json)

      with open(path_join(fake_gt_root, '%s_fake_walls.json' % floor_name), 'w') as f:
        json.dump(fake_wall_json, f)

      with open(path_join(fake_gt_root, '%s_fake_doors.json' % floor_name), 'w') as f:
        json.dump(fake_door_json, f)

      with open(path_join(fake_gt_root, '%s_fake_columns.json' % floor_name), 'w') as f:
        json.dump(fake_column_json, f)
