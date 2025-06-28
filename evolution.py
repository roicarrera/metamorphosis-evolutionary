import numpy as np
import random
from copy import deepcopy
import cv2 as cv
import os
import matplotlib.pyplot as plt

from Butterfly import Butterfly
from Shape import Shape, ShapeName
from butterfly_generator import generate_butterfly_img, generate_random_shape, generate_random_butterflies

global MAX_INDIVIDUALS
MAX_INDIVIDUALS = 300

def evolve(n, butterflies, objective):
    max_fitness = []
    avg_fitness = []
    for i in range(n):
        evaluated_butterflies = evaluate(butterflies, objective)
        selected_butterflies = select_fittest(evaluated_butterflies, 0.1)
        crossover_butterflies = crossover(selected_butterflies, 0.60)
        mutated_butterflies = mutate(crossover_butterflies, 0.7)
        max_fitness.append(max([x[1] for x in mutated_butterflies]))
        avg_fitness.append(sum(x[1] for x in mutated_butterflies) / len(mutated_butterflies))
        butterflies = refill(mutated_butterflies)
        print(f'Finished iteration {i} of {n}')
    final_evaluation = evaluate(butterflies, objective)
    final_evaluation = sorted(final_evaluation, key=lambda item: item[1], reverse=True)
    for i in range(10):
        path = os.path.join('images', f"final_result{i}.png")
        result_img = generate_butterfly_img(final_evaluation[i][0])
        print(f'Evaluation of butterfly number {i}: {final_evaluation[i][1]}')
        cv.imwrite(path, result_img)
    plt.plot(avg_fitness, color='green', label='Average fitness')
    plt.plot(max_fitness, color='red', label='Max fitness')
    plt.show()

def fitness_function(objective: Butterfly, candidate: Butterfly):
    objective_f = objective.astype(np.float32)
    candidate_f = candidate.astype(np.float32)
    mse = -np.mean((objective_f - candidate_f) ** 2)
    return mse

def evaluate(butterflies: list[Butterfly], objective: Butterfly):
    results = []
    for butterfly in butterflies:
        butterfly_img = generate_butterfly_img(butterfly)
        objective_img = generate_butterfly_img(objective)
        result = fitness_function(objective_img, butterfly_img)
        results.append((butterfly, result))
    return results

def select_fittest(butterflies: list[(Butterfly, float)], retain_ratio: float):
    fittest = sorted(butterflies, key=lambda item: item[1], reverse=True)[:int(len(butterflies) * retain_ratio)]
    return fittest

def crossover(butterflies: list[(Butterfly, float)], new_ratio: float):
    for _ in range(int(MAX_INDIVIDUALS * new_ratio - len(butterflies))):
        b1, b2 = random.sample(butterflies, 2)
        contour = max([b1, b2], key=lambda x: x[1])[0].contour
        symmetry = max([b1, b2], key=lambda x: x[1])[0].symmetry
        bg_color = (b1[0].bg_color + b2[0].bg_color) // 2
        shapes = crossover_shapes(b1[0].shapes, b2[0].shapes)
        b3 = Butterfly(contour, symmetry, bg_color, shapes)
        butterflies.append((b3, 0.0))
    return butterflies

def crossover_shapes(shapes1: list[Shape], shapes2: list[Shape]) -> list[Shape]:
    child_shapes = []
    len_target = min(10, max(len(shapes1), len(shapes2)))
    
    for i in range(len_target):
        source = None
        if i < len(shapes1) and i < len(shapes2):
            source = random.choice([shapes1[i], shapes2[i]])
        elif i < len(shapes1):
            source = shapes1[i]
        elif i < len(shapes2):
            source = shapes2[i]
        else:
            break
        child_shapes.append(deepcopy(source))
    
    if len(child_shapes) > 0 and random.random() < 0.2:
        idx = random.randint(0, len(child_shapes)-1)
        child_shapes[idx] = generate_random_shape()

    return child_shapes

def mutate(butterflies: list[(Butterfly, float)], mutation_rate: float):
    for butterfly in butterflies:
        if random.random() < mutation_rate and butterfly[1] != 0:
            if random.random() < mutation_rate:
                butterfly[0].contour = np.random.randint(1,9)
            if random.random() < mutation_rate:
                butterfly[0].symmetry = bool(np.random.randint(0,2))
            if random.random() < mutation_rate:
                butterfly[0].bg_color =  min(255, max(100, butterfly[0].bg_color + random.randint(-30, 30)))
            if random.random() < mutation_rate:
                butterfly[0].shapes = mutate_shapes(butterfly[0].shapes, mutation_rate)
    return butterflies

def mutate_shape(shape: Shape, mutation_rate: float) -> Shape:
    new_shape = deepcopy(shape)
    
    if random.random() < mutation_rate:
        new_shape.shape_type = random.choice(list(ShapeName))

    if random.random() < mutation_rate:
        new_shape.intensity = min(255, max(0, new_shape.intensity + random.randint(-30, 30)))

    if random.random() < mutation_rate:
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)
        new_shape.coordinates = (new_shape.coordinates[0] + dx, new_shape.coordinates[1] + dy)

    if random.random() < mutation_rate:
        new_shape.size = max(0.1, min(0.5, new_shape.size + random.uniform(-0.1, 0.1)))

    return new_shape


def mutate_shapes(shapes: list[Shape], mutation_rate: float) -> list[Shape]:
    mutated = [mutate_shape(s, mutation_rate) for s in shapes]

    if random.random() < mutation_rate:
        if random.random() < 0.5 and len(mutated) > 1:
            mutated.pop(random.randint(0, len(mutated)-1))
        else:
            mutated.append(generate_random_shape())

    return mutated

def refill(butterflies_with_fitness: list[tuple[Butterfly, float]]) -> list[Butterfly]:
    butterflies = [bf[0] for bf in butterflies_with_fitness]
    
    if len(butterflies) < MAX_INDIVIDUALS:
        needed = MAX_INDIVIDUALS - len(butterflies)
        new_butterflies = generate_random_butterflies(needed)
        butterflies.extend(new_butterflies)

    return butterflies
