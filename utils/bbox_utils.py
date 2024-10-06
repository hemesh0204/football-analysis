import math

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2)//2, (y1 + y2)//2

def get_bbox_wdth(bbox):
    return bbox[2] - bbox[0]

def measure_distance(p1, p2):

    value1 = (p1[0] - p2[0])**2
    value2 = (p1[1] - p2[1])**2
    
    return math.sqrt(value1 + value2)

def measure_xy_distance(p1, p2):
    return p1[0]-p2[0], p1[0] - p2[1]


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2)/2), int(y2)