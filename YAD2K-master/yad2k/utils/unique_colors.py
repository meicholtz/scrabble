import colorsys
import random


def unique_colors(N):
    '''Get a list of unique colors.

    Positional arguments:
        N       number of colors to compute

    Returns:
        colors  list of N unique colors

    Example:

    '''
    hsv_tuples = [(x / N, 1., 1.) for x in range(N)]

    # Convert HSV tuples to RGB
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

    # Convert float to int [0-255]
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    # Shuffle colors to decorrelate adjacent classes
    random.seed(10101)  # fix seed for consistency
    random.shuffle(colors)
    random.seed(None)  # reset seed

    return colors
