import config
import utils

import logging
import numpy as np

from scipy.spatial import ConvexHull
from typing import Tuple


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
        subjectPolygon: a list of (x,y) 2d points, any polygon.
        clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
        **points have to be counter-clockwise ordered**

    Return:
        a list of (x,y) vertex point for the intersection polygon.
    """
    def inside(p):
        return(cp2[0]-cp1[0])*(p[1]-cp1[1]) >= (cp2[1]-cp1[1])*(p[0]-cp1[0])
    
    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]] 
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0] 
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
    
    outputList = subjectPolygon
    cp1 = clipPolygon[-1]
    
    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]
    
        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    # 1D implementation: 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return 0.5 * np.abs(np.sum(x * np.roll(y, 1, axis=-1), axis=-1) - np.sum(y * np.roll(x, 1, axis=-1), axis=-1))

def convex_hull_intersection(p1, p2):
    result = np.zeros((p1.shape[0], p2.shape[0]))
    for i in range(len(p1)):
        for j in range(len(p2)):
            inter_p = polygon_clip(p1[i], p2[j])
            if inter_p is not None:
                hull_inter = ConvexHull(inter_p, qhull_options="QJ Pp")
                result[i, j] = hull_inter.volume
    return result

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def counter_clockwise(rectangle: np.ndarray) -> np.ndarray:
    x, y = rectangle[..., 0], rectangle[..., 1]
    filter = np.sum(x * np.roll(y, 1, axis=-1), axis=-1) - np.sum(y * np.roll(x, 1, axis=-1), axis=-1) > 0
    rectangle[filter] = rectangle[filter][:, ::-1]
    return rectangle

def volume(corners) -> float:
    width = np.sqrt(np.sum(np.power(corners[..., 0, :2] - corners[..., 1, :2], 2), axis=1))
    length = np.sqrt(np.sum(np.power(corners[..., 0, :2] - corners[..., 3, :2], 2), axis=1))
    height = np.abs(corners[..., 0, 2] - corners[..., 4, 2])
    return (width * length * height)

@utils.profile(output_root="profiling", enabled=config.debug)
def iou_batch(
    ground: np.ndarray,
    target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """ Compute 3D bounding box IoU.

    Parameters:
        ground: numpy array (n,8,3), assume up direction is Z
        target: numpy array (n,8,3), assume up direction is Z
    Return:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    """
    ground = ground.astype(np.float64)
    target = target.astype(np.float64)

    minimal = np.min([ground.min(), target.min(), .0])
    ground -= minimal
    target -= minimal
    logging.debug(f"Minimal: {minimal}")

    logging.debug(f"Identical: {np.array_equal(ground, target)}")

    n: int = np.min([30, ground.shape[0], target.shape[0]])
    gface = ground[:, :4][..., :2]
    logging.debug(f"Ground Face [{gface.shape}, {gface.dtype}]: \n{gface[:n]}")
    tface = target[:, :4][..., :2]
    logging.debug(f"Target Face [{tface.shape}, {tface.dtype}]: \n{tface[:n]}")
    gtdiff = gface[:n] - tface[:n]
    logging.debug(f"Ground-Target difference [mean: {gtdiff.mean(0)}]: \n{gtdiff}\n...")
    
    gface = counter_clockwise(gface)
    logging.debug(f"Ground Face clockwise: {[is_clockwise(face) for face in gface[:n]]}")
    tface = counter_clockwise(tface)
    logging.debug(f"Target Face clockwise: {[is_clockwise(face) for face in tface[:n]]}")
    
    garea = poly_area(gface[..., 0], gface[..., 1])
    logging.debug(f"Area1 [{garea.shape}, {garea.dtype}]: \n{garea[:n]}\n...")
    tarea = poly_area(tface[..., 0], tface[..., 1])
    logging.debug(f"Area2 [{tarea.shape}, {tarea.dtype}]: \n{tarea[:n]}\n...")

    inter_area = convex_hull_intersection(gface, tface)
    logging.debug(f"Inter Area (initial) [{inter_area.shape}, {inter_area.dtype}]: \n{inter_area[:n]}\n...")

    intersection = np.argwhere(inter_area > 0)
    rows, cols = intersection[:, 0], intersection[:, 1]
    logging.debug(f"Intersection [{intersection.shape}]: \n{intersection[:n]}\n...")

    inter_area = inter_area[rows, cols]
    logging.debug(f"Inter Area [{inter_area.shape}, {inter_area.dtype}]: \n{inter_area[:n]}\n...")
    
    iou_2d = inter_area / (garea[rows] + tarea[cols] - inter_area)
    logging.debug(f"2D IoU  [{iou_2d.shape}, {iou_2d.dtype}]: \n{iou_2d[:n]}\n...")

    ground_filtered = ground[rows]
    target_filtered = target[cols]

    zmax = np.min([ground_filtered[..., 4, 2], target_filtered[..., 4, 2]], axis=0)
    zmin = np.max([ground_filtered[..., 0, 2], target_filtered[..., 0, 2]], axis=0)
    difference = np.max([np.full_like(zmax, .0), zmax - zmin], axis=0)
    logging.debug(f"Difference [{difference.shape}, {difference.dtype}]: \n{difference[:n]}\n...")

    inter_vol = inter_area * difference
    logging.debug(f"Inter Volume [{inter_vol.shape}, {inter_vol.dtype}]: \n{inter_vol[:n]}\n...")

    vol1 = np.array([ConvexHull(pts, qhull_options="QJ Pp").volume for pts in ground_filtered])
    logging.debug(f"Ground Volume [{vol1.shape}, {vol1.dtype}]: \n{vol1[:n]}\n...")
    vol2 = np.array([ConvexHull(pts, qhull_options="QJ Pp").volume for pts in target_filtered])
    logging.debug(f"Target Volume [{vol2.shape}, {vol2.dtype}]: \n{vol2[:n]}\n...")

    iou_3d = np.zeros((ground.shape[0], target.shape[0]), dtype=np.float32)
    iou_3d[rows, cols] = inter_vol / (vol1 + vol2 - inter_vol)
    iou_3d[iou_3d > 1.1] = 0
    logging.debug(f"3D IoU: \n{iou_3d}")
    return iou_3d, iou_2d
