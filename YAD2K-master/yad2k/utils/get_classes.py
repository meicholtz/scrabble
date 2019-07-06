def get_classes(filename):
    '''Load classes from text file.

    Positional arguments:
        filename    string indicating path to text file

    Returns:
        classes     list of strings for each class

    Notes:
    - the expected format for the input text file is one class per line

    Example:
    - read animal classes from file:

    $ cat animals.txt
    dog
    cat
    turtle
    horse

    $ python
    >>> import utils
    >>> classes = utils.get_classes("animals.txt")
    >>> print(classes)
    ['dog', 'cat', 'turtle', 'horse']
    '''
    with open(filename) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    return classes
