from butterfly_generator import *
import evolution
import os

global MAX_INDIVIDUALS
MAX_INDIVIDUALS = 1000

if __name__ == '__main__':
    butterflies = generate_random_butterflies(MAX_INDIVIDUALS)
    #objective = generate_random_butterflies(1)
    #objective_img = generate_butterfly_img(objective[0])
    #path = os.path.join('images', f"result.png")
    #cv.imwrite(path, objective_img)
    objective_img = cv.imread('images/owl.png', 0)
    mask = cv.imread(os.path.join('images', f"mask5.png"), 0)
    objective_img = cv.bitwise_and(objective_img, mask)
    cv.imwrite('images/objective.png', objective_img)
    evolution.evolve(300, butterflies, objective_img, MAX_INDIVIDUALS)