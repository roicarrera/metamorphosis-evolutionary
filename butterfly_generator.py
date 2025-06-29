import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import random

from Butterfly import Butterfly
from Shape import Shape, ShapeName

IMG_WIDTH = 256
IMG_HEIGHT = 256

def generate_butterfly_img(butterfly):
    base_img = np.full((IMG_WIDTH, IMG_HEIGHT, 3), butterfly.bg_color, dtype=np.uint8)

    for shape in butterfly.shapes:
        if shape.shape_type == ShapeName.CIRCLE:
            cv.circle(base_img, shape.coordinates, int(shape.size * IMG_HEIGHT / 2), shape.intensity, -1)
        elif shape.shape_type == ShapeName.SQUARE:
            side = int(shape.size * IMG_HEIGHT)
            top_left = (int(shape.coordinates[0] - side/2), int(shape.coordinates[1] - side/2))
            bottom_right = (int(shape.coordinates[0] + side/2), int(shape.coordinates[1] + side/2))
            cv.rectangle(base_img, top_left, bottom_right, color=shape.intensity, thickness=-1)
        elif shape.shape_type == ShapeName.TRIANGLE:
            side = int(shape.size * IMG_HEIGHT)
            height = np.sqrt(side**2 - (side/2)**2)
            bottom_left = (int(shape.coordinates[0] - side/2), int(shape.coordinates[1] + height/3))
            top = (int(shape.coordinates[0]), int(shape.coordinates[1] - 2*height/3))
            bottom_right = (int(shape.coordinates[0] + side/2), int(shape.coordinates[1] + height/3))
            points = np.array([bottom_left, top, bottom_right], np.int64)
            cv.fillPoly(base_img, [points], color=shape.intensity)
        elif shape.shape_type == ShapeName.PENTAGON:
            side = int(shape.size * IMG_HEIGHT)
            r = side / (2 * np.sin(np.pi / 5))
            cx, cy = shape.coordinates
            angle_offset = -np.pi / 2
            points = []
            for i in range(5):
                angle = angle_offset + 2 * np.pi * i / 5
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                points.append((x, y))

            points_array = np.array(points, np.int64)
            cv.fillPoly(base_img, [points_array], color=shape.intensity)
    if butterfly.symmetry:
        left_side = base_img[:, :int(IMG_WIDTH/2)]
        right_side = np.fliplr(left_side)
        base_img = np.concatenate((left_side, right_side), axis=1)

    mask = cv.imread(os.path.join('images', f"mask{butterfly.contour}.png"))
    butterfly_img = cv.bitwise_and(base_img, mask)
    return butterfly_img

def generate_random_butterflies(n):
    butterflies = []
    for _ in range(n):
        number_of_shapes = np.random.randint(1, 16)
        shapes = []
        for _ in range(number_of_shapes):
            shape = generate_random_shape()
            shapes.append(shape)
        butterfly = Butterfly(np.random.randint(1,9), bool(np.random.randint(0,2)), (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)), shapes)
        butterflies.append(butterfly)
    return butterflies

def generate_random_shape():
    return Shape(random.choice(list(ShapeName)), (np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)), (np.random.randint(0, 257), np.random.randint(0, 257)), np.random.random()/2)
