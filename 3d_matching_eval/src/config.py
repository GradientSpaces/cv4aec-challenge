from typing import List

""" Enable Numba & CUDA optimized algorithms.
    All of the algorithms are defined in `calculations.py`."""
enable_optimization: bool = True

""" Normalize user's predictions with ground-truth no matter the initial vertices parameters are.
    Normalization includes rotation, scale, translation, ratios approximation.
    In the result data outputted all the differences will be specified."""
enable_normalization: bool = False

""" Metrics (precision, recall, f1) and 3D IoU thresholds specified in ground-truth units.
    The maximum is also used as LAP threshold value where the L2 norm is the default measurement."""
metrics_thresholds: List[float] = [0.05, 0.10, 0.20]
iou_thresholds: List[float] = [0.25, 0.50, 0.75]

""" Debug."""
debug: bool = False
