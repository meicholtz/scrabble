import numpy as np


def get_anchors(filename):
    '''Load anchors from text file.

    Positional arguments:
        filename    string indicating path to text file

    Returns:
        anchors     numpy ndarray of anchors

    Notes:
    - The expected format for the input text file is one anchor per line
    - The output shape will be (m, n), where m is the number of anchors and n is the dimensionality

    Example:
    - Read sample 2D anchors from file:

    $ cat model_data/yolo_anchors.txt
    0.57273 0.677385
    1.87446 2.06253
    3.33843 5.47434
    7.88282 3.52778
    9.77052 9.16828

    $ python
    >>> import utils
    >>> anchors = utils.get_anchors("model_data/yolo_anchors.txt")
    >>> print(anchors)
    [[0.57273  0.677385]
     [1.87446  2.06253 ]
     [3.33843  5.47434 ]
     [7.88282  3.52778 ]
     [9.77052  9.16828 ]]
    >>> type(anchors)
    <class 'numpy.ndarray'>
    >>> anchors.shape
    (5, 2)
    '''
    anchors = np.loadtxt(filename)
    return anchors
