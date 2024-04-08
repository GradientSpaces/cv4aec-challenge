## OBJ File Format
The floorplan is available in both a binary obj format and a plain text JSON format. For the OBJ format, the header contains following information:

```
number of layers (unsigned int)
number of structures for each layer (unsigned int array)
<Followed by information for layer 1>
 1. length of layer name (unsigned int)
 2. layer name (bytes array)
 3. information for layer 1/structure 1:
  a. number of points for structure (unsigned int)
  b. pt1.X, pt1.Y, pt2.X, pt2.Y, ... (float in meters)
 4. information for layer 1/structure 2:
 ...
<Followed by information for layer 2>:
...
```

The detailed file format is provided here:
```
[number of layers][number of structures for each layer]
[length of layer 1's name][layer 1 name]
[number of points for structure 1][pt1.X][pt1.Y][pt2.X][pt2.Y]...
[number of points for structure 2][pt1.X][pt1.Y][pt2.X][pt2.Y]...
...
[length of layer 2's name][layer 2 name]
[number of points for structure 1][pt1.X][pt1.Y][pt2.X][pt2.Y]...
[number of points for structure 2][pt1.X][pt1.Y][pt2.X][pt2.Y]...
...
```

## Sample OBJ file
A sample obj file is given below (it should be in binary practically):
```
2 32 15
6 "A_WALL"
2 320.12 442.55 320.12 445.55
2 322.12 447.55 322.12 449.55
...
6 "A_DOOR"
4 178.12 336.55 225.12 482.55 389.12 557.55 175.12 882.55
2 389.12 557.55 175.12 882.55
```
## JSON File Format
```
1. number of layers
2. number of structures in each layer
3. layer details
 - layer name
 - number of points
 - x y coordinates for each point
```
### Geometric Metrics :

- IoU of the each room (a room is defined as a completely separated area with walls and doors).
- Accuracy of endpoints. Precision/Recall at 3 different thresholds: 5cm, 10cm and 20cm, as well as the F-measure at each threshold will be evaluated in the coordinate system of the point cloud. The provided endpoints will be matched with the Hungarian algorithm to the point cloud, and every point that is within a certain threshold will be determined as a match.
- Orientation. For each matched line between the ground truth, we will compute the cosine similarity metric between them as the normalized dot product. If a line is not matched with ground truth, the cosine metric will be zero. Finally, the metric will be averaged over all the ground truth lines. 

### Topological Metrics :
- Warping error([Jain et al. 2010](https://ieeexplore.ieee.org/document/5539950)) : The warping error will first warp the predicted floorplan to the ground truth with a homotopic deformation, and then compute the pixels that cannot match after the deformation.
- Betti number error :The Betti number error will compare the Betti numbers between the prediction and the ground truth and output the absolute value of the difference. 
