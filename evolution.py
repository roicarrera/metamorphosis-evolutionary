import numpy as np
import random
from copy import deepcopy
import cv2 as cv
import os
import matplotlib.pyplot as plt

from Butterfly import Butterfly
from Shape import Shape, ShapeName
from butterfly_generator import generate_butterfly_img, generate_random_shape, generate_random_butterflies

def evolve(n, butterflies, objective, max_individuals):
    global MAX_INDIVIDUALS
    MAX_INDIVIDUALS = max_individuals
    max_fitness = []
    avg_fitness = []
    mix_butterflies = []
    for i in range(n):
        retain_ratio = 0.2
        new_individuals_ratio = 0.80
        mutation_ratio = 0.7
        evaluated_butterflies = evaluate(butterflies, objective)
        selected_butterflies = select_fittest(evaluated_butterflies, retain_ratio)
        mix_butterflies.append(generate_butterfly_img(selected_butterflies[0][0]))
        crossover_butterflies = crossover(selected_butterflies, new_individuals_ratio, mutation_ratio)
        max_fitness.append(max([x[1] for x in crossover_butterflies if x[1]]))
        avg_fitness.append(sum(x[1] for x in crossover_butterflies if x[1]) / len([x for x in crossover_butterflies if x[1]]))
        butterflies = refill(crossover_butterflies)
        print(f'Finished iteration {i} of {n}. Max fitness: {max_fitness[-1]}')
    final_evaluation = evaluate(butterflies, objective)
    final_evaluation = sorted(final_evaluation, key=lambda item: item[1], reverse=True)
    for i in range(10):
        path = os.path.join('images', f"final_result{i}.png")
        result_img = generate_butterfly_img(final_evaluation[i][0])
        print(f'Evaluation of butterfly number {i}: {final_evaluation[i][1]}')
        cv.imwrite(path, result_img)
    save_video(mix_butterflies, "evolution.mp4", fps=5.0)
    plt.plot(avg_fitness, color='green', label='Average fitness')
    plt.plot(max_fitness, color='red', label='Max fitness')
    plt.show()

def save_video(frames: list[np.ndarray], output_path: str, fps: float = 2.0):
    height, width = frames[0].shape
    out = cv.VideoWriter(
        output_path,
        cv.VideoWriter_fourcc(*'XVID'),  # Prueba también con 'MJPG'
        fps,
        (width, height),
        isColor=True 
    )

    for img in frames:
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR) 
        out.write(img)

    out.release()

def fitness_function(objective: Butterfly, candidate: Butterfly):
    objective_f = objective.astype(np.float32)
    candidate_f = candidate.astype(np.float32)
    mse = -np.mean((objective_f - candidate_f) ** 2)
    return mse

def evaluate(butterflies: list[Butterfly], objective):
    if isinstance(objective, Butterfly):
        objective_img = generate_butterfly_img(objective)
    else:
        objective_img = objective
    results = []
    for butterfly in butterflies:
        butterfly_img = generate_butterfly_img(butterfly)
        result = fitness_function(objective_img, butterfly_img)
        results.append((butterfly, result))
    return results

def select_fittest(butterflies: list[(Butterfly, float)], retain_ratio: float):
    fittest = sorted(butterflies, key=lambda item: item[1], reverse=True)[:int(len(butterflies) * retain_ratio)]
    return fittest

def crossover(butterflies: list[(Butterfly, float)], new_ratio: float, mutation_rate: float):
    new_butterflies = []
    for _ in range(int(MAX_INDIVIDUALS * new_ratio - len(butterflies))):
        b1, b2 = random.sample(butterflies, 2)
        contour = max([b1, b2], key=lambda x: x[1])[0].contour
        symmetry = max([b1, b2], key=lambda x: x[1])[0].symmetry
        bg_color = (b1[0].bg_color + b2[0].bg_color) // 2
        shapes = crossover_shapes(b1[0].shapes, b2[0].shapes)
        b3 = Butterfly(contour, symmetry, bg_color, shapes)
        new_butterflies.append(mutate([(b3, None)], mutation_rate)[0])
    butterflies.extend(new_butterflies)
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
        elif len(mutated) < 10:
            mutated.append(generate_random_shape())

    return mutated

def refill(butterflies_with_fitness: list[tuple[Butterfly, float]]) -> list[Butterfly]:
    butterflies = [bf[0] for bf in butterflies_with_fitness]
    
    if len(butterflies) < MAX_INDIVIDUALS:
        needed = MAX_INDIVIDUALS - len(butterflies)
        new_butterflies = generate_random_butterflies(needed)
        butterflies.extend(new_butterflies)

    return butterflies
