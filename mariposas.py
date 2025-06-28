import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

from Butterfly import Butterfly
from Shape import Shape, ShapeName

IMG_WIDTH = 256
IMG_HEIGHT = 256

def generate_butterfly_img(butterfly):
    base_img = np.ones([IMG_WIDTH, IMG_HEIGHT], dtype=np.uint8) * butterfly.bg_color

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
            points = np.array([bottom_left, top, bottom_right], np.int64).reshape(-1, 1, 2)
            cv.fillPoly(base_img, [points], shape.intensity)
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

            points_array = np.array(points, np.int64).reshape(-1, 1, 2)
            cv.fillPoly(base_img, [points_array], shape.intensity)
    if butterfly.symmetry:
        left_side = base_img[:, :int(IMG_WIDTH/2)]
        right_side = np.fliplr(left_side)
        base_img = np.concatenate((left_side, right_side), axis=1)

    mask = cv.imread(os.path.join('images', f"mask{butterfly.contour}.png"), 0)
    butterfly_img = cv.bitwise_and(base_img, mask)

    plt.imshow(butterfly_img, cmap='gray', vmin=0, vmax=255)
    plt.show()
    return butterfly_img

if __name__ == '__main__':
    shape = Shape(shape_type=ShapeName.PENTAGON, intensity=80, coordinates=(160, 160), size=0.2)
    shape2 = Shape(shape_type=ShapeName.TRIANGLE, intensity=150, coordinates=(160, 160), size=0.1)
    butterfly = Butterfly(contour=1, symmetry=False, bg_color=128, shapes=[shape, shape2])
    generate_butterfly_img(butterfly)
