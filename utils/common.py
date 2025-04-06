import math


def euclidean_distance(first_point: list, second_point: list):
    op_1 = (second_point[0] - first_point[0])**2
    op_2 = (second_point[1] - first_point[1])**2
    return math.sqrt(op_1 + op_2)