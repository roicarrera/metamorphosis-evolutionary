from Shape import Shape

class Butterfly():
    def __init__(self, contour: int, symmetry: bool, bg_color: tuple[int, int, int], shapes: list[Shape]):
        self.contour = contour
        self.symmetry = symmetry
        self.bg_color = bg_color
        self.shapes = shapes
