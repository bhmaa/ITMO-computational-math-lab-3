import math
import os
import random
import re
import sys


def first_function(args) -> float:
    return math.sin(args[0])


def second_function(args) -> float:
    return (args[0] * args[1]) / 2


def third_function(args) -> float:
    return pow(args[0], 2) * pow(args[1], 2) - 3 * pow(args[0], 3) - 6 * pow(args[1], 3) + 8


def fourth_function(args) -> float:
    return pow(args[0], 4) - 9 * args[1] + 2


def fifth_function(args) -> float:
    return args[0] + pow(args[0], 2) - 2 * args[1] * args[2] - 0.1


def six_function(args) -> float:
    return args[1] + pow(args[1], 2) + 3 * args[0] * args[2] + 0.2


def seven_function(args) -> float:
    return args[2] + pow(args[2], 2) + 2 * args[0] * args[1] - 0.3


def default_function(args) -> float:
    return 0.0

def get_functions(n: int):
    if n == 1:
        return [first_function, second_function]
    elif n == 2:
        return [third_function, fourth_function]
    elif n == 3:
        return [fifth_function, six_function, seven_function]
    else:
        return [default_function]

def partial_derivative(args, function, unknown_id, h = 1e-5):
    args_h = args.copy()
    args_h[unknown_id] += h
    return (function(args_h) - function(args)) / h


def solve_by_fixed_point_iterations(system_id, number_of_unknowns, initial_approximations):
    system = get_functions(system_id)
    current_values = initial_approximations
    next_values = [0.0] * number_of_unknowns
    for iteration in range(100000):
        for i in range(number_of_unknowns):
            partial_deriv = partial_derivative(current_values, system[i], i)
            if abs(partial_deriv) < 1e-10: 
                next_values[i] = current_values[i]
            else:
                next_values[i] = current_values[i] - system[i](current_values) / partial_deriv
        is_convergent = all(abs(next_values[i] - current_values[i]) < 0.00001 for i in range(number_of_unknowns))
        if is_convergent:
            return next_values
        current_values = next_values
    raise RuntimeError("Maximum iterations reached without convergence.")

if __name__ == '__main__':
    system_id = int(input().strip())

    number_of_unknowns = int(input().strip())

    initial_approximations = []

    for _ in range(number_of_unknowns):
        initial_approximations_item = float(input().strip())
        initial_approximations.append(initial_approximations_item)

    result = solve_by_fixed_point_iterations(system_id, number_of_unknowns, initial_approximations)

    print('\n'.join(map(str, result)))
