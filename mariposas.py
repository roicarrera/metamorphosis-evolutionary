import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

from Butterfly import Butterfly
from Shape import Shape, ShapeName

IMG_WIDTH = 256
IMG_HEIGHT = 256

def generate_butterfly(butterfly):
    base_img = np.ones([IMG_WIDTH, IMG_HEIGHT], dtype=np.uint8) * butterfly.bg_color

    for shape in butterfly.shapes:
        if shape.shape_type == ShapeName.CIRCLE:
            cv.circle(base_img, shape.coordinates, int(shape.size * IMG_HEIGHT / 2), shape.intensity, -1)

    mask = cv.imread(os.path.join('images', f"mask{butterfly.contour}.png"), 0)
    butterfly_img = cv.bitwise_and(base_img, mask)

    plt.imshow(butterfly_img, cmap='gray', vmin=0, vmax=255)
    plt.show()
    return

if __name__ == '__main__':
    shape = Shape(shape_type=ShapeName.CIRCLE, intensity=80, coordinates=(160, 160), size=0.1)
    butterfly = Butterfly(contour=1, symmetry=False, bg_color=128, shapes=[shape])
    generate_butterfly(butterfly)
