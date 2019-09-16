import ipdb
import numpy as np


def get_data(filename):
    '''Load images and bounding boxes from numpy data file. Works for 2D and 3D images.

    Positional arguments:
        filename    string indicating path to npz file

    Returns:
        images      ndarray of images; shape=(n, h, w, [d], b), where n->number of images,
                    h->image height, w->image width, d->image depth (for 3D only), b->batch; dtype=float64
        boxes       ndarray of bounding boxes; shape=(n, m, s), where n->number of images,
                    m->maximum number of boxes in any image, s->shape of bounding box (s=5 for 2D, s=7 for 3D); dtype=float64

    Notes:
    - The input npz file is expected to contain the following variables (i.e. keys): images, boxes

    Example:
    - Read sample images containing 2D shapes from file:

    $ python
    >>> import utils
    >>> images, boxes = utils.get_data("images/shapes.npz")
    >>> images.dtype
    dtype('float64')
    >>> boxes.dtype
    dtype('float64')
    '''

    data = np.load(filename, allow_pickle=True)  # load data from file
    # Process images
    images = data['images'] / 255.  # convert images from uint8 to float64
    image_size = np.array(images.shape[1:])  # ignore images.shape[0] (number of images)
    image_size = np.expand_dims(image_size, axis=0)  # make row vector

    images = np.expand_dims(images, axis=len(images.shape))  # add dimension for batches

    # Process bounding boxes
    boxes = data['boxes']
    assert len(set([box.shape[1] for box in boxes])) == 1, "All boxes must have same dimensions"

    # Compute position and shape (in pixels) of every box in every image
    # Removed normalization of boxes because my text files are already normalized - Alexander Faus
    if boxes[0].shape[1] == 5:  # 2D case
        box_position = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        box_size = [(box[:, 3:5] - box[:, 1:3]) for box in boxes]
    else:  # 3D case
        box_position = [0.5 * (box[:, 4:7] + box[:, 1:4]) / image_size for box in boxes]
        box_size = [(box[:, 4:7] - box[:, 1:4]) / image_size for box in boxes]

    # Concatenate boxes into [x, y, (z), w, h, (d), class], where z and d are only used in 3D
    boxes = [np.concatenate((box_position[i], box_size[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

    # Find the maximum number of boxes in any one image, then pad all instances with fewer boxes   (for training purposes)
    max_box_shape = max([box.shape for box in boxes])
    for i, box in enumerate(boxes):
        boxes[i] = np.zeros(max_box_shape, dtype=boxes[0].dtype)  # array of zeros (for padding)
        boxes[i][:box.shape[0], :box.shape[1]] = box
    boxes = np.array(boxes)  # convert to ndarray (as opposed to a list of ndarrays)
    return images, boxes
