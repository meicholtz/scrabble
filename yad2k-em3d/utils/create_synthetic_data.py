import numpy as np
import pymrt.geometry as geo


def gen_syn_images(num_train_images, max_obj_per_img, im_size):
    assert len(im_size) == 2

    obj_funcs = [geo.circle, geo.square]
    # Reference calls:
    #   circ = geo.circle(shape=(416,416),radius=40,position=0.5)
    #   sq = geo.square(shape=(416,416),side=40,position=0.5)
    # TODO: Random size adjustment
    obj_params = [{'radius': 40}, {'side': 40}]
    # this is kinda hacky but gives a functional way to easily get the size we should add to generate bounding box
    obj_sizes = [lambda x: x['radius'], lambda x: x['side']]

    # Generate shape w/pymrt
    # -- train script expects 0->255 input range
    image_data_arr = np.empty((num_train_images, im_size[0], im_size[1]), dtype=np.uint8)

    all_bboxes = []
    for im_idx in range(num_train_images):
        boxes = []
        num_obj = np.random.randint(1, max_obj_per_img + 1)  # returns [1,requested_high)
        for obj_idx in range(num_obj):
            obj_type = np.random.randint(len(obj_funcs))  # equal balance of types now...more sophisiticated sampling needed
            params = dict(obj_params[obj_type])  # get any default params for this obj and build from there
            params['shape'] = (im_size[0], im_size[1])
            params['position'] = tuple(np.random.rand(2))  # completely random position (0,1): does not ensure overlap
            obj = obj_funcs[obj_type](**params)  # call object function with parameters to generate shape image

            # Add bounding box in expected yad2k format: [class, x_min, y_min, x_max, y_max]
            boxes.append([obj_type, np.int(np.round(params['shape'][0] * params['position'][0])),
                          np.int(np.round(params['shape'][1] * params['position'][1])),
                          np.int(np.round((params['shape'][0] + obj_sizes[obj_type](params)) * params['position'][0])),
                          np.int(np.round((params['shape'][1] + obj_sizes[obj_type](params)) * params['position'][1]))])

            # TODO: check bounding box creation and make sure rounding works right
            image_data_arr[im_idx, :, :] = image_data_arr[im_idx, :, :] + obj * 255
        all_bboxes.append(np.array(boxes))

    return (image_data_arr, all_bboxes)


if __name__ == '__main__':
    im_save_dir = 'syn_images'   # set to '' or None to disable saving generated imgs to file
    out_np_path = 'syn_data.npz'  # path to save numpy file in yad2k with generated images
    show_img = False             # requires X-forwarding while running script
    num_train_images = 5
    max_obj_per_img = 4
    im_size = (416, 416)

    (image_arr, bboxes) = gen_syn_images(num_train_images, max_obj_per_img, im_size)

    if out_np_path:
        np.savez(out_np_path, images=image_arr, boxes=bboxes)

    if im_save_dir:
        import PIL
        import os
        os.makedirs(im_save_dir, exist_ok=True)
        for im_idx, out_image in enumerate(image_arr):
            im = PIL.Image.fromarray(out_image)
            im.save(os.path.join(im_save_dir, 'gen_%d.png' % (im_idx)))
            if show_img:  # Somewhat buggy but opens up the files eventually
                im.show()
