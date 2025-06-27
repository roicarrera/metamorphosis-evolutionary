from enum import Enum

class ShapeName(Enum):
    CIRCLE = 'circle'
    SQUARE = 'square'
    TRIANGLE = 'triangle'
    PENTAGON = 'pentagon'

class Shape():
    def __init__(self, shape_type: ShapeName, intensity: int, coordinates: tuple[int, int], size: float):
        self.shape_type = shape_type
        self.intensity = intensity
        self.coordinates = coordinates
        self.size = size