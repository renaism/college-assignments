from math import sin, exp, cos, sqrt, pi
from decimal import Decimal
import math
import random

# The range limits (design space) of which the function is tested
X1_MIN, X1_MAX = -10, 10
X2_MIN, X2_MAX = -10, 10

# Minimum temperature until the cycle is stopped
T_MIN = 10e-6

def f(x1, x2):
    """
    Description : The original function of the problem. The result is always negative.
    Parameter   : x1 [Decimal], x2 [Decimal]
    Return      : f(x1, x2) (Decimal)
    """
    return Decimal(-abs(sin(x1) * cos(x2) * exp(abs(1 - (sqrt(math.pow(x1, 2) + math.pow(x2, 2) ) / pi)))))

def g(x1 , x2):
    """
    Description : The negative of the original function f. Contrary to the original function, the result is always positive.
    Parameter   : x1 [Decimal], x2 [Decimal]
    Return      : g(x1, x2) = -f(x1, x2) [Decimal]
    """
    return (-f(x1, x2))

def get_random_point():
    """
    Description : Get a uniformly distributed random point (x1, x2) in the design space.
    Return      : x1 [Decimal], x2 [Decimal]
    """
    return Decimal(random.uniform(X1_MIN, X1_MAX)), Decimal(random.uniform(X1_MIN, X2_MAX))

def get_initial_temp(n):
    """
    Description : Get the average value of g evaluated at n randomly selected point(s) in the design space.
    Parameter   : n [integer]
    Return      : initial temperature [Decimal]
    """
    sum = 0
    for _ in range(n):
        x1, x2 = get_random_point()
        sum += g(x1, x2)
    return Decimal(sum / n)

def sa(n, c):
    """
    Description : Find the minimum value of f(x1, x2) from the maximum value of g(x1, x2) with simulated annealing.
    Parameter   : n (iterations/cycle) [integer], c (reduction factor) [Decimal]
    Return      : estimated minimum value of f(x1, x2) [Decimal]
    """
    # Generate the starting temperature
    t = get_initial_temp(100)
    print("Initial temperature:", round(t, 3))
    # Select the initial design point
    x1_c, x2_c = get_random_point()
    f_c = g(x1_c, x2_c)
    print("Initial design point:", round(x1_c, 5), round(x2_c, 5))
    # Variable for storing the maximum value that found during the entire simulation
    g_maximum = {
        "points": (x1_c, x2_c),
        "value": f_c
    }
    # Set the cycle number
    p = 1
    # Cycle will keep running until the temperature fall under T_MIN
    while t > T_MIN:
        for _ in range(n):
            # Generate new design point in the design space
            x1_i, x2_i = get_random_point()
            f_i = g(x1_i, x2_i)
            delta_f = f_i - f_c
            # If the value of the new design point is lower than the previous, evaluate using Metropolis criterion
            if delta_f < 0:
                # Boltzmann's probability distribution
                p_i = exp(delta_f/t)
                # If the random number is lower, accept the new design point as the next design point
                if (random.random() < p_i):
                    accept = True
                else:
                    accept = False
            # If the value of the new design point is greater than the previous, accept it as the next design point
            else:
                accept = True
            if accept:
                # Make the new design point as the next design point
                x1_c, x2_c = x1_i, x2_i
                f_c = g(x1_c, x2_c)
                # If the value of the new design point is greater than current maximum, set it as maximum
                if f_c > g_maximum["value"]:
                    g_maximum["value"] = f_c
                    g_maximum["points"] = (x1_c, x2_c)
        """print("Cycle:", p, "Temperature:", round(t, 3), "Max of g(x1, x2):", round(maximum, 5))"""
        # Update the cycle number
        p += 1
        # Reduce the temperature
        t *= c
    # Return the negative of the maximum (Since g is the negative of f)
    f_minimum = {
        "points": g_maximum["points"],
        "value": -g_maximum["value"]
    }
    return f_minimum

# Input n (iterations/cycle) and c (reduction factor)
# Will keep asking for input until the input criteria is satisfied (n > 0 and 0 < c < 1)
n, c = -1, -1
while n <= 0 or c <= 0 or c >= 1:
    try:
        n, c = input("[n] [c] >> ").split()
        n = int(n)
        c = Decimal(c)
    except ValueError:
        n, c = -1, -1

# Run the simulated annealing and display the result
result = sa(n, c)
print("---Result---")
print("x1, x2 = {}, {}".format(round(result["points"][0], 5), round(result["points"][1], 5)))
print("f(x1, x2) = {}".format(round(result["value"], 5)))