import cv2
import numpy as np

from typing import List, Tuple, Union

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def normalize(
    endpoints: Union[np.ndarray, List[np.ndarray]],
    structures: Union[np.ndarray, List[np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    for i in range(len(endpoints)):
        endpoints[i] = endpoints[i].reshape(-1, 3)

    lead: int = np.argmax([[np.mean(endpoints_.max(0) - endpoints_.min(0)) for endpoints_ in endpoints]])
    indices = list(range(len(endpoints)))
    indices.remove(lead) 

    min, divisor = endpoints[lead].min(), (endpoints[lead].max() - endpoints[lead].min())
    for i in range(len(endpoints)):
        endpoints[i] = (endpoints[i] - min) / divisor
    
    mean = endpoints[lead].mean(0)
    for i in range(len(endpoints)):
        endpoints[i] = endpoints[i] - mean
    
    max = endpoints[lead].max()
    for i in range(len(endpoints)):
        endpoints[i] = endpoints[i] / max
    
    for i in range(len(endpoints)):
        endpoints[i] = endpoints[i].reshape(-1, 8, 3)

    """ TODO: Normalize structures.
    centers = structures[:, :6].reshape(-1, 3)
    centers = (centers - centers.min()) / (centers.max() - centers.min())
    centers = centers - centers.mean(0)
    centers = centers / centers.max()
    structures[:, :6] = centers.reshape(-1, 6)"""
    return endpoints, structures

def plot(
    endpoints: Union[np.ndarray, List[np.ndarray]],
    structures: Union[np.ndarray, List[np.ndarray]],
    width: int = 780, height: int = 780, depth: int = 780,
    origin: np.ndarray = None,
    model: str = None, floor: str = None
) -> np.ndarray:
    """ Plot stuctures from the endpoints."""
    endpoints, structures = normalize(endpoints, structures)

    padding: float = 0.8
    scale: np.ndarray = np.array([width, height, depth], dtype=np.int32) * padding / 2
    margin: np.ndarray = (np.array([width, height, depth]) / 2).astype(np.int32)

    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]

    origin: np.ndarray = origin if origin is not None else np.full((height, width, 3), 255, dtype=np.uint8)
    for i in range(len(endpoints)):
        for index in range(endpoints[i].shape[0]):
            # Plot bbox endpoints.
            coordinates = endpoints[i][index, :4]
            coordinates = coordinates * scale + margin
            tl, bl, br, tr = coordinates[:, :2].astype(int)

            cv2.line(origin, tl, tr, colors[i], 1)
            cv2.line(origin, bl, br, colors[i], 1)
            cv2.line(origin, tl, bl, colors[i], 1)
            cv2.line(origin, br, tr, colors[i], 1)

            """ TODO: Plot structures middle line.
            (x1, y1, _) = (structures[index, :3] * scale + margin).astype(int)
            (x2, y2, _) = (structures[index, 3:6] * scale + margin).astype(int)
            drawline(origin, (x1, y1), (x2, y2), (0, 0, 0), 1, gap=5)"""
    if model or floor:
        text: str = f"Model: {model or 'Unknown'} - {floor or 'Unknown'}"
        cv2.putText(origin, text, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
    return origin