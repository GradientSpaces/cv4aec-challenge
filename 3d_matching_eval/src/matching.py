import cv2
import numpy as np
import logging

from pathlib import Path
from collections import defaultdict
from functools import partial
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple

import config
import loader
import utils

from calculations.cost_matrix import calculate_cost_matrix
from calculations.iou import iou_batch
from calculations.rigid_registration import RigidRegistration


@utils.profile(output_root="profiling", enabled=config.debug)
def align(gt_normalized: np.ndarray, tg_normalized: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """ Returns alignment `scale` factor, `rotation` matrix and `translation` matrix to transform
        target matrix `tg_normalized` to `gt_normalized`."""
    logging.debug("Using RigidRegistration implementation to calculate the matrix...")
    def print_iter(iteration, error, X, Y):
        print(f"[RigidRegistration] Matching iter: {iteration}, error: {error:.2f}", end="\r")

    callback = partial(print_iter)
    reg = RigidRegistration(**{'X': gt_normalized, 'Y': tg_normalized})
    # TODO: Speed up `register` with Numba.
    # TODO: !IMPORTANT Find a measurable error of the alignment (`q`, ...).
    TY, (scale, rotation, translation) = reg.register(callback)
    print()
    return (scale, rotation, translation)

def calculate_metrics(
    gtindex: Dict[str, np.ndarray],
    gtendpoints: np.ndarray,
    tgindex: Dict[str, np.ndarray],
    tgendpoints: np.ndarray
) -> Dict:
    metrics: Dict = defaultdict(dict)

    for key in gtindex.keys():
        metrics[key] = defaultdict(dict)

        gtsample = gtendpoints[gtindex[key]]
        gtsample = gtsample.reshape(-1, 3)
        metrics[key]["total"] = gtsample.shape[0]

        tgsample = tgendpoints[tgindex.get(key, np.array([]))]
        tgsample = tgsample.reshape(-1, 3)
        metrics[key]["predicted"] = tgsample.shape[0]

        matches: Dict[List] = defaultdict(list)
        if len(tgsample) == 0:
            logging.warning(f"Cannot find '{key}' class in the target data for metrics matching.")
            logging.info(f"Available keys in ground data: {gtindex.keys()}")
            logging.info(f"Available keys in target data: {tgindex.keys()}")
            for threshold in config.metrics_thresholds:
                matches[threshold] = []
        else:
            """ Returns linear assignment problem solution using scipy Hungarian implementation.
                Cost function between verticies is defined in `calculate_cost_matrix` method.
                Default: Euclidean distance (L2)."""
            logging.debug(f"Calculating metrics for '{key}': {gtsample.shape[0]} over {tgsample.shape[0]} structures... Optimization: {config.enable_optimization}")
            cost_matrix = calculate_cost_matrix(gtsample, tgsample)

            logging.debug("Applying linear_sum_assignment...")
            rows, cols = linear_sum_assignment(cost_matrix, maximize=False)
            for row, col in list(zip(rows, cols)):
                distance: float = cost_matrix[row, col]
                for threshold in config.metrics_thresholds:
                    if distance < threshold:
                        matches[threshold].append(distance)
        
        metrics[key]["thresholds"] = {}
        for threshold, matched in matches.items():
            # Calculate metrics according to `compute_precision_recall_helper`:
            # https://github.com/seravee08/WarpingError_Floorplan/blob/main/IOU_precision_recall/ipynb/main.ipynb
            if tgsample.shape[0] != 0:
                precision: float = len(matched) / tgsample.shape[0]
                recall: float = len(matched) / gtsample.shape[0]
                f1: float = (2 * precision * recall) / (precision + recall)
            else:
                precision: float = .0
                recall: float = .0
                f1: float = .0
            metrics[key]["thresholds"][threshold] = {
                "matched": len(matched),
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        metrics[key] = dict(metrics[key])
    return dict(metrics)

def calculate_iou(
    gtindex: Dict[str, np.ndarray],
    gtendpoints: np.ndarray,
    tgindex: Dict[str, np.ndarray],
    tgendpoints: np.ndarray
) -> Dict[str, np.ndarray]:
    ious: Dict[str, List] = defaultdict(list)

    """ Calculate IoU based on GT classes only."""
    for key in gtindex.keys():
        gtsample = gtendpoints[gtindex[key]]
        tgsample = tgendpoints[tgindex.get(key, np.array([]))]

        if tgsample.shape[0] != 0:
            logging.debug(f"Calculating 3D IoU for '{key}': {gtsample.shape[0]} over {tgsample.shape[0]} structures...")
            iou3d = iou_batch(gtsample, tgsample)[0]
        else:
            logging.warning(f"Cannot find '{key}' class in the target data for IoU matching.")
            logging.info(f"Available keys in ground data: {gtindex.keys()}")
            logging.info(f"Available keys in target data: {tgindex.keys()}")
            iou3d = np.array([[]])

        logging.debug("Applying linear_sum_assignment...")
        rows, cols = linear_sum_assignment(iou3d, maximize=True)
        for row, col in list(zip(rows, cols)):
            ious[key].append(iou3d[row, col])
        
        lacklen: int = len(gtsample) - len(tgsample)
        if lacklen > 0:
            ious[key].extend([.0] * lacklen)
        else:
            logging.warn(f"In classname '{key}' sample lack length is: {lacklen}")
        ious[key] = np.asarray(ious[key], dtype=np.float32)
        
    general = {}
    for classname, iou in ious.items():
        general[classname] = {
            "min": iou.min(),
            "max": iou.max(),
            "mean": iou.mean(),
            "median": np.median(iou),
            "std": iou.std()
        }
    ious["general"] = general
    return ious

@utils.profile(output_root="profiling", enabled=config.debug)
def match(
    gtstructures: np.ndarray,
    tgstructures: np.ndarray,
    output: Path, model: str, floor: str
) -> Dict:
    np.set_printoptions(precision=4, suppress=True)

    gtindex, gtendpoints = loader.read_endpoints(gtstructures)
    tgindex, tgendpoints = loader.read_endpoints(tgstructures)

    """ Reshape to flatten arrays for point-cloud alignment."""
    ground = gtendpoints.reshape(-1, 3)
    logging.info(f"Ground endpoints: mean {ground.mean():.2f}, max {ground.max(0)}, "
                    f"min {ground.min(0)}, size: {(ground.max(0) - ground.min(0))} "
                    f"(avg: {np.mean((ground.max(0) - ground.min(0))):.2f})")

    target = tgendpoints.reshape(-1, 3)
    logging.info(f"Target endpoints: len  mean {target.mean():.2f}, max {target.max(0)}, "
                    f"min {target.min(0)}, size: {(target.max(0) - target.min(0))} "
                    f"(avg: {np.mean((target.max(0) - target.min(0))):.2f})")

    if config.enable_normalization:
        """ Use Coherent Point Drift Algorithm for preprocessing alignment.
            Source: https://github.com/siavashk/pycpd."""
        scale, rotation, translation = align(ground, target)
        logging.debug(f"Alignment scale ratio: {scale:.4f}")
        logging.debug(f" - rotation: \n{rotation}")
        logging.debug(f" - translation: \n{translation}")

        """Matricies alignment formula:"""
        translation = -np.dot(np.mean(target, 0), rotation) + translation + np.mean(ground, 0)
        target = np.dot(target, rotation) + translation
        target *= scale
        target = target.astype(np.float32)

        logging.debug(f"Aligned target endpoints: mean {target.mean():.2f}, max {target.max(0)}, "
            f"min {target.min(0)}, size: {(target.max(0) - target.min(0))} "
            f"(avg: {np.mean((target.max(0) - target.min(0))):.2f})")
    else:
        scale, rotation, translation = None, None, None

    ground = ground.reshape(-1, 8, 3)
    target = target.reshape(-1, 8, 3)

    width, height = 1024, 1024
    origin = utils.plot([ground, target], [gtstructures, tgstructures], width, height, model=model, floor=floor)

    """ Calculate base metrics: precision, recall."""
    metrics = calculate_metrics(gtindex, ground, tgindex, target)
    for classname, values in metrics.items():
        logging.info(f"Classname: {classname}")
        logging.info(f"  Metrics: {values}")

    """ Calculate 3D IoU grouped by classname: walls, collumns, doors."""
    ious = calculate_iou(gtindex, ground, tgindex, target)
    for classname, iou in ious.items():
        if classname == "general":
            continue
        logging.info(f"Classname: {classname}")
        logging.info(f"  IoU: min {iou.min():.4f}, max {iou.max():.4f}, mean {iou.mean():.4f}, median {np.median(iou):.4f}, std {np.std(iou):.4f}")

    results = {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "metrics": metrics,
        "ious": ious,
    }

    if config.debug:
        results["config"] = {
            "enable_optimization": config.enable_optimization,
            "enable_normalization": config.enable_normalization,
            "metrics_thresholds": config.metrics_thresholds,
            "iou_thresholds": config.iou_thresholds,
            "debug": config.debug
        }

    logging.debug("Showing preview data using OpenCV...")
    cv2.imwrite(str(output.joinpath(f"{model}-{floor}.jpg")), origin)
    return results
