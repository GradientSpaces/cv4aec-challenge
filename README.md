# Scan-to-BIM Competition 2024
We, the <a href="https://gradientspaces.stanford.edu/">Gradient Spaces group</a> at Stanford University, together with collaborators from Oregon State University and ETH Zurich are hosting the 4th International Scan-to-BIM challenge.  Our challenge is hosted at the workshop on <a href="https://cv4aec.github.io">Computer Vision In The Built Environment For The Design, Construction and Operation of Buildings</a> in conjunction with <a href="https://cvpr.thecvf.com/">The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2024</a>. To this end, the challenge will include the following tasks:

1. 2D Floorplan Reconstruction
2. 3D Building Model Reconstruction

## Data Download Instructions
Links to Download Data:

### 2D Floorplan Reconstruction
- [Training Data: Point Cloud + Ground Truth](https://oregonstate.box.com/s/kwjn01g84d0fka2j2vrjfd8j8w5n37e6)
- [Validation Data: Point Cloud + Ground Truth](https://oregonstate.box.com/s/ber73d1njhxo4vwc4i4qmtbmyyimrprj)
- [Testing Data: Point Cloud ONLY](https://oregonstate.box.com/s/op8bul06ea1hm7ldnfmqecrt32kctk64)

### 3D Building Model Reconstruction
- [Training Data: Point Cloud + Ground Truth](https://oregonstate.box.com/s/lozaa1tvcwk119hnfg8ftv23xlcxxyuy)
- [Testing Data: Point Cloud ONLY](https://oregonstate.box.com/s/fhuzl8lfi1sxa4e0rw3w7ynw0erdsicx)

## Submission
The submission should be made in the same JSON format as in the provided ground truth. The codalab submission portal will open on **18 April, 2024**. Links will be updated on the website!

> We would like to note that ALL the submissions **need to be constructed automatically** . Manual reconstructions are against the spirit of this challenge and will not be allowed.

## About the Data

### 2D Floorplan Reconstruction
The 2D Floorplan Reconstruction challenge contains a total of 31 buildings with multiple floors each and dozens of rooms on each floor. Of which, 20 buildings are designated as the training set, with a total of 49 point clouds. The validation and testing sets contain 5.5 buildings with 21 point clouds each. For each model, there is an aligned point cloud in LAZ format. For the training and validation sets, a corresponding floorplan aligned with the coordinate system of the point cloud is also provided. 

### 3D Building Model Reconstruction
The training data consists of 16 floors from 8 buildings. For each model, there is an aligned point cloud in LAZ format. The 3D building coordinates for walls, columns and doors are presented in 3 separate JSON files. We focus on the reconstruction of walls, columns, and doors.

## Evaluation

### 2D Floorplan Reconstruction
We include metrics to evaluate the reconstruction of the walls, doors, and columns, as well as floor area in 2D : 

1. **Geometric Metrics** \
    a. _IoU_ of each room (a room is defined as a completely separated area with walls and doors). \
    b. _Accuracy of endpoints_ : Precision/Recall at 3 different thresholds: 5cm, 10cm and 20cm, as well as the F-measure at each threshold will be evaluated in the coordinate system of the point cloud. The provided endpoints will be matched with the Hungarian algorithm to the point cloud, and every point that is within a certain threshold will be determined as a match. \
    c. _Orientation_ For each matched line between the ground truth, we will compute the cosine similarity metric between them as the normalized dot product. If a line is not matched with ground truth, the cosine metric will be zero. Finally, the metric will be averaged over all the ground truth lines.

2. **Topological Metrics** \
    a. _[Warping error](https://ieeexplore.ieee.org/document/5539950)_ : The warping error will first warp the predicted floorplan to the ground truth with a homotopic deformation, and then compute the pixels that cannot match after the deformation. \
    b. **_Betti number error_** : The Betti number error will compare the Betti numbers between the prediction and the ground truth and output the absolute value of the difference.

> For exact details and code regarding evaluation metrics, please look here: [2d_floorplan_eval](https://github.com/GradientSpaces/cv4aec-challenge/tree/main/2d_floorplan_eval)

### 3D Building Model Reconstruction
We evaluate the submissions on a variety of metrics : 

1. **3D IoU** of the 3D bounding box of each wall
2. **Accuracy of the endpoints** : Precision/Recall at 3 different thresholds: 5cm, 10cm and 20cm, as well as F-measure will be evaluated in the coordinate system of the point cloud. The provided endpoints will be matched with the Hungarian algorithm to the point cloud, and every point that is within a certain threshold will be determined as a match. We evaluate per each of the three semantic types (i.e., wall, column, door).

> For exact details and code regarding evaluation metrics, please look here: [3d_matching_eval](https://github.com/GradientSpaces/cv4aec-challenge/tree/main/3d_matching_eval)

For more details, please refer to the challenge website: https://cv4aec.github.io/
