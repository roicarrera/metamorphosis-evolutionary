from butterfly_generator import *
import evolution
import os

global MAX_INDIVIDUALS
MAX_INDIVIDUALS = 100

if __name__ == '__main__':
    butterflies = generate_random_butterflies(MAX_INDIVIDUALS)
    objective = generate_random_butterflies(1)
    objective_img = generate_butterfly_img(objective[0])
    path = os.path.join('images', f"result.png")
    cv.imwrite(path, objective_img)
    evolution.evolve(100, butterflies, objective[0])