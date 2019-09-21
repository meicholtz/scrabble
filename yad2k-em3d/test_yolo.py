#! /usr/bin/env python
"""Test a 2D YOLO network (v2) on images."""
import argparse
import imghdr
import os

import ipdb
import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import scipy.io

from models.keras_yolo import create_model, yolo_eval, yolo_head
import utils

parser = argparse.ArgumentParser(description="Test a 2D YOLO network (v2) on images.")
parser.add_argument(
    '-m','--model_path',
    help='path to h5 file containing trained YOLO network')
parser.add_argument(
    '-d', '--data_path',
    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images'")
parser.add_argument(
    '-a', '--anchors_path',
    help='path to anchors file, defaults to model_data/yolo_anchors.txt',
    default=os.path.join('model_data', 'yolo_anchors.txt'))
parser.add_argument(
    '-c', '--classes_path',
    help='path to classes file, defaults to model_data/shape_classes.txt',
    default=os.path.join('model_data', 'shape_classes.txt'))
parser.add_argument(
    '-o', '--output_path',
    help='path to output test results (as a mat file), defaults to testing/temp.mat',
    default=os.path.join('testing', 'temp.mat'))
parser.add_argument(
    '-b', '--batch',
    help='batch size for testing images, defaults to 2',
    default=2)


def _main(args):
    # Parse input arguments
    model_path = os.path.expanduser(args.model_path)
    data_path = os.path.expanduser(args.data_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)
    output_path = os.path.expanduser(args.output_path)

    batch = args.batch
    assert model_path.endswith('.h5'), 'model_path must have .h5 extension'
    assert output_path.endswith('.mat'), 'output_path must have .mat extension'

    # Extract anchors and classes from input files
    anchors = utils.get_anchors(anchors_path)
    classes = utils.get_classes(classes_path)
    images, boxes = utils.get_data(data_path)

    # Create model and load weights from file
    model_body, model = create_model(images.shape[1:-1], int(boxes.shape[-1]), anchors, classes)
    model_body.load_weights(model_path)
    model.summary()
    ipdb.set_trace()

    # Pass input data through the network in batches
    output = model_body.predict(images[0:batch, :, :, :])
    for i in range(batch, images.shape[0], batch):
        output = np.concatenate((output, model_body.predict(images[i:i + batch, :, :, :])))

    # Save output file
    if output_path != '':
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scipy.io.savemat(output_path, mdict={'output': output})
    print('Results saved to file: {}'.format(output_path))

    # # Generate colors for drawing bounding boxes
    # colors = utils.unique_colors(len(classes))

    # ipdb.set_trace()
    # Generate output tensor targets for filtered bounding boxes
    # TODO: Wrap these backend operations with Keras layers
    # yolo_outputs = yolo_head(model_body.output, anchors, len(classes))
    # input_image_shape = K.placeholder(shape=(2, ))
    # boxes, scores, classes_outputs = yolo_eval(yolo_outputs, input_image_shape, score_threshold=thold0, iou_threshold=thold1)

    # sess = K.get_session()  # TODO: Remove dependence on Tensorflow session

    # if output_path != '':
    #     # os.makedirs(output_path, exist_ok=True)
    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # for image_file in os.listdir(data_path):
    #     try:
    #         image_type = imghdr.what(os.path.join(data_path, image_file))
    #         if not image_type:
    #             continue
    #     except IsADirectoryError:
    #         continue

    #     image = Image.open(os.path.join(data_path, image_file))
    #     is_fixed_size = True  # temporary hack!
    #     if is_fixed_size:  # TODO: When resizing we can use minibatch input
    #         resized_image = image.resize(tuple(reversed((416, 416))), Image.BICUBIC)
    #         image_data = np.array(resized_image, dtype='float32')
    #     else:
    #         # Due to skip connection + max pooling in YOLO_v2, inputs must have
    #         # width and height as multiples of 32.
    #         new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
    #         resized_image = image.resize(new_image_size, Image.BICUBIC)
    #         image_data = np.array(resized_image, dtype='float32')
    #         print(image_data.shape)
    #     image_data /= 255.
    #     if image.mode == 'L':
    #         image_data = np.expand_dims(image_data, -1)  # add channel dimension
    #         image = image.convert('RGB')
    #     image_data = np.expand_dims(image_data, 0)  # add batch dimension

    #     out_boxes, out_scores, out_classes = sess.run(
    #         [boxes, scores, classes],
    #         feed_dict={
    #             model_body.input: image_data,
    #             input_image_shape: [image.size[1], image.size[0]],
    #             K.learning_phase(): 0})
    #     print('Found {} boxes for {}'.format(len(out_boxes), image_file))

    #     font = ImageFont.truetype(
    #         font='font/FiraMono-Medium.otf',
    #         size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    #     thickness = (image.size[0] + image.size[1]) // 300

    #     for i, c in reversed(list(enumerate(out_classes))):
    #         predicted_class = classes[c]
    #         box = out_boxes[i]
    #         score = out_scores[i]

    #         label = '{} {:.2f}'.format(predicted_class, score)

    #         draw = ImageDraw.Draw(image)
    #         label_size = draw.textsize(label, font)

    #         top, left, bottom, right = box
    #         top = max(0, np.floor(top + 0.5).astype('int32'))
    #         left = max(0, np.floor(left + 0.5).astype('int32'))
    #         bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    #         right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    #         print(label, (left, top), (right, bottom))

    #         if top - label_size[1] >= 0:
    #             text_origin = np.array([left, top - label_size[1]])
    #         else:
    #             text_origin = np.array([left, top + 1])

    #         # My kingdom for a good redistributable image drawing library.
    #         for i in range(thickness):
    #             draw.rectangle(
    #                 [left + i, top + i, right - i, bottom - i],
    #                 outline=colors[c])
    #         draw.rectangle(
    #             [tuple(text_origin), tuple(text_origin + label_size)],
    #             fill=colors[c])
    #         draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    #         del draw
    #     out_file_name = '.'.join(os.path.basename(image_file).split('.')[:-1]) + '.txt'
    #     image.save(os.path.join(output_path, image_file), quality=90)
    #     bfo = open(os.path.join(output_path, out_file_name), 'w')
    #     for box, score, cla in zip(out_boxes, out_scores, out_classes):
    #         line_list = [cla, score] + [x for x in box]
    #         line_strs = [x.__str__() for x in line_list]
    #         bfo.write(' '.join(line_strs) + '\n')
    #     bfo.close()
    # sess.close()


if __name__ == '__main__':
    _main(parser.parse_args())
