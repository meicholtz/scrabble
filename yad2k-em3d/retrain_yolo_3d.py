"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import argparse
import os

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from keras import backend as K
from keras.layers import Input, Lambda, Conv3D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from yad2k.models.keras_yolo_3d import (
    preprocess_true_boxes, yolo_body, yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

# Args
argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images'",
    default=os.path.join('..', 'DATA', 'underwater_data.npz'))

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default=os.path.join('model_data', 'yolo_anchors.txt'))

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default=os.path.join('..', 'DATA', 'underwater_classes.txt'))

argparser.add_argument(
    '-o',
    '--output_prefix',
    help='save prefix for resulting models, defaults to train1',
    default='train1')

# Default anchor boxes
YOLO_ANCHORS = np.array((
    (0.57273, 0.677385, 0.57273), (1.87446, 2.06253, 1.87446),
    (3.33843, 5.47434, 3.3383), (7.88282, 3.52778, 7.88282),
    (9.77052, 9.16828, 9.77052)))


def _main(args):
    # Parse input arguments
    save_prefix = args.output_prefix
    data_path = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)

    # Extract classes and anchors from text files
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    anchors = YOLO_ANCHORS  # overrides the anchors_path for the time being

    # Load training data from numpy file
    #   (1) an array of 'images'
    #   (2) an object array 'boxes' (variable length of boxes in each image)
    data = np.load(data_path)

    image_data, boxes = process_data(data['images'], data['boxes'], im_size=args.image_size)

    detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors, im_size=args.image_size)

    model_body, model = create_model(anchors, class_names, load_pretrained=False, save_prefix=save_prefix, im_size=args.image_size)

    model.summary()

    train(
        model,
        class_names,
        anchors,
        image_data,
        boxes,
        detectors_mask,
        matching_true_boxes,
        save_prefix=save_prefix
    )

    draw(model_body,
         class_names,
         anchors,
         image_data,
         image_set='val',  # assumes training/validation split is 0.9
         weights_name=os.path.join(save_prefix, 'trained_stage_3_best.h5'),
         out_path=os.path.join(save_prefix, 'output_images'),
         save_all=False)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 3)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS


def process_data(images, boxes=None, im_size=(416, 416, 416)):
    '''processes the data'''
    im_size0 = images.shape
    if im_size0[1:] == im_size:
        processed_images = images / 255.
        orig_size = np.array(im_size0[1:])
        orig_size = np.expand_dims(orig_size, axis=0)
    else:
        images = [PIL.Image.fromarray(i) for i in images]
        orig_size = np.array([images[0].width, images[0].height])
        orig_size = np.expand_dims(orig_size, axis=0)

        # Image preprocessing
        processed_images = [i.resize(im_size, PIL.Image.BICUBIC) for i in images]
        processed_images = [np.array(image, dtype=np.float) for image in processed_images]
        processed_images = [image / 255. for image in processed_images]

    if boxes is not None:
        # Box preprocessing
        # Get extents as y_min, x_min, z_min, y_max, x_max, z_max, class for comparision with model output.
        boxes_extents = [box[:, [2, 1, 3, 5, 4, 6, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xyz = [0.5 * (box[:, 4:7] + box[:, 1:4]) for box in boxes]
        boxes_whd = [box[:, 4:7] - box[:, 1:4] for box in boxes]
        boxes_xyz = [boxxyz / orig_size for boxxyz in boxes_xyz]
        boxes_whd = [boxwhd / orig_size for boxwhd in boxes_whd]
        boxes = [np.concatenate((boxes_xyz[i], boxes_whd[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0] < max_boxes:
                zero_padding = np.zeros((max_boxes - boxz.shape[0], 7), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        out_images = np.array(processed_images)
        if len(out_images.shape) < 4:
            out_images = out_images[:, :, :, None]

        return out_images, np.array(boxes)
    else:
        out_images = np.array(processed_images)
        if len(out_images.shape) < 4:
            out_images = out_images[:, :, :, None]

        return out_images


def get_detector_mask(boxes, anchors, im_size=(416, 416, 416)):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, list(im_size))

    return np.array(detectors_mask), np.array(matching_true_boxes)


def create_model(anchors, class_names, load_pretrained=True, freeze_body=True, in_channels=1, save_prefix='', im_size=(416, 416, 416)):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (im_size[0] / 32, im_size[0] / 32, 5, 1)
    matching_boxes_shape = (im_size[0] / 32, im_size[0] / 32, 5, 5)

    # Create model input layers.
    image_input = Input(shape=im_size + (in_channels,))
    boxes_input = Input(shape=(None, 7))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        os.makedirs(os.path.join(save_prefix, 'model_data'), exist_ok=True)
        # Save topless yolo:
        topless_yolo_path = os.path.join(save_prefix, 'model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join(save_prefix, 'model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv3D(len(anchors) * (7 + len(class_names)), (1, 1, 1), activation='linear')(topless_yolo.output)  # *

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model


def train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes, validation_split=0.1, save_prefix=''):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
    input_shape = tuple(model.input_shape[0][1:3])
    if save_prefix != '':
        os.makedirs(save_prefix, exist_ok=True)
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    logging = TensorBoard()
    checkpoint = ModelCheckpoint(os.path.join(save_prefix, "trained_stage_3_best.h5"), monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=validation_split,
              batch_size=32,
              epochs=5,
              callbacks=[logging])
    model.save_weights(os.path.join(save_prefix, 'trained_stage_1.h5'))

    model_body, model = create_model(anchors, class_names, load_pretrained=False, freeze_body=False, im_size=input_shape)

    model.load_weights(os.path.join(save_prefix, 'trained_stage_1.h5'))

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=0.1,
              batch_size=8,
              epochs=30,
              callbacks=[logging])

    model.save_weights(os.path.join(save_prefix, 'trained_stage_2.h5'))

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=0.1,
              batch_size=8,
              epochs=30,
              callbacks=[logging, checkpoint, early_stopping])

    model.save_weights(os.path.join(save_prefix, 'trained_stage_3.h5'))


def draw(model_body, class_names, anchors, image_data, image_set='val',
         weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''
    if image_set == 'train':
        image_data = np.array([np.expand_dims(image, axis=0)
                               for image in image_data[:int(len(image_data) * .9)]])
    elif image_set == 'val':
        image_data = np.array([np.expand_dims(image, axis=0)
                               for image in image_data[int(len(image_data) * .9):]])
    elif image_set == 'all':
        image_data = np.array([np.expand_dims(image, axis=0)
                               for image in image_data])
    else:
        ValueError("draw argument image_set must be 'train', 'val', or 'all'")
    # model.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0.0)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                      class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path, str(i) + '.png'))

        # To display (pauses the program):
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()


if __name__ == '__main__':
    args = argparser.parse_args()
    args.image_size = (160, 160, 96)
    _main(args)
