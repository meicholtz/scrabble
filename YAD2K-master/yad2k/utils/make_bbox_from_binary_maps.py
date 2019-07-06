import skimage.io
import skimage.measure
import numpy as np

def conv_to_bbox(binary_map_path):
    """ Read a binary tiff volume and put each tiff image's 1 regions into a bounding box list
    
    Keyword arguments:
    binary_map_path -- path to multi-page tiff image volume

    Returns: [] of []'s of numpy arrays with bounding boxes in form [class, x_min, y_min, x_max, y_max]
    """
    data = skimage.io.MultiImage(binary_map_path)
    # Note, marking all boxes as class 0

    bboxes = []
    for page_img in data:
        page_img = page_img[:,:768]
        label_img = skimage.measure.label(page_img)
        props = skimage.measure.regionprops(label_img)
        # From skimage regionprops bbox: Bounding box (min_row, min_col, max_row, max_col).
        boxes = []
        for obj in props:
            boxes.append([0,obj.bbox[1],obj.bbox[0],obj.bbox[3],obj.bbox[2]])
        bboxes.append(np.array(boxes)) 
    return bboxes

def read_im_pages_to_list(im_tiff_path):
    # Returns a list of images read from multi-page tiff
    data = skimage.io.MultiImage(im_tiff_path)
    im_arr_list = [x[:,:768] for x in data]
    return im_arr_list


epfl_training_im = '/home/hdr/data/epfl/training.tif'
epfl_training_gt = '/home/hdr/data/epfl/training_groundtruth.tif'
epfl_training_save = '/home/hdr/data/epfl/training_w_bbox_crop.npz'

epfl_testing_im = '/home/hdr/data/epfl/testing.tif'
epfl_testing_gt = '/home/hdr/data/epfl/testing_groundtruth.tif'
epfl_testing_save = '/home/hdr/data/epfl/testing_w_bbox_crop.npz'

ims = read_im_pages_to_list(epfl_training_im)
bbox = conv_to_bbox(epfl_training_gt)
import pdb; pdb.set_trace()

np.savez(epfl_training_save,images=ims,boxes=bbox)

ims = read_im_pages_to_list(epfl_testing_im)
bbox = conv_to_bbox(epfl_testing_gt)

np.savez(epfl_testing_save,images=ims,boxes=bbox)

