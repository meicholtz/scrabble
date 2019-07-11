"""3D YOLO model (v2) defined in Keras."""
import os

import ipdb
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Input, Lambda, Conv3D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam

from .keras_darknet19_3d import DarknetConv3D, DarknetConv3D_BN_Leaky, darknet_body_3d
import utils


def create_model(image_size, box_size, anchors, classes):
    '''Create YOLO model.

    Positional arguments:
        image_size      tuple of input image size (h, w, d)
        box_size        output bounding box size; type=int (should be constant, =7 for 3D)
        anchors         numpy.ndarray of anchor boxes used to predict bounding boxes
        classes         list of class names (str)

    returns:
        model_body      YOLOv2 with new output layer
        model           YOLOv2 with custom loss Lambda layer
    '''

    # Parse inputs
    num_anchors = len(anchors)
    num_classes = len(classes)

    scale = 32  # integer downsampling factor for determining grid size, i.e., how many pixels/cell
    grid_size = tuple([int(x / scale) for x in image_size])  # output grid size

    # Create model input layers
    image_input = Input(shape=image_size + (1,))
    boxes_input = Input(shape=(None, box_size))
    detectors_mask_input = Input(shape=grid_size + (num_anchors, 1))
    matching_boxes_input = Input(shape=grid_size + (num_anchors, box_size))

    # Create model body
    yolo_model = yolo_body(image_input, box_size, num_anchors, num_classes)
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    final_layer = Conv3D(num_anchors * (box_size + num_classes), (1, 1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes}
        )([model_body.output, boxes_input, detectors_mask_input, matching_boxes_input])

    model = Model([model_body.input, boxes_input, detectors_mask_input, matching_boxes_input], model_loss)

    return model_body, model


def get_detector_mask(boxes, anchors, image_size):
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
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, list(image_size))

    return np.array(detectors_mask), np.array(matching_true_boxes)


def preprocess_true_boxes(true_boxes, anchors, image_size):
    """Find detector in YOLO where ground truth box should appear.

    Parameters
    ----------
    true_boxes : array
        List of ground truth boxes in form of relative x, y, z, w, h, d, class.
        Relative coordinates are in the range [0, 1] indicating a percentage
        of the original image dimensions.
    anchors : array
        List of anchors in form of w, h, d.
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.
    image_size : array-like
        List of image dimensions in form of h, w, d in pixels. # check order

    Returns
    -------
    detectors_mask : array
        0/1 mask for detectors in [conv_height, conv_width, conv_depth, num_anchors, 1]
        that should be compared with a matching ground truth box.
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
    """
    height, width, depth = image_size
    num_anchors = len(anchors)
    # Downsampling factor of 5x 2-stride max_pools == 32.
    # TODO: Remove hardcoding of downscaling calculations.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert depth % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    conv_height = height // 32
    conv_width = width // 32
    conv_depth = depth // 32
    num_box_params = true_boxes.shape[1]
    detectors_mask = np.zeros(
        (conv_height, conv_width, conv_depth, num_anchors, 1), dtype=np.float32)
    matching_true_boxes = np.zeros(
        (conv_height, conv_width, conv_depth, num_anchors, num_box_params),
        dtype=np.float32)

    for box in true_boxes:
        # scale box to convolutional feature spatial dimensions
        box_class = box[-1]
        box = box[0:6] * np.array(
            [conv_width, conv_height, conv_depth, conv_width, conv_height, conv_depth])
        i = np.floor(box[1]).astype('int')
        j = np.floor(box[0]).astype('int')
        k = np.floor(box[2]).astype('int')
        best_iou = 0
        best_anchor = 0
        for anchor_idx, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box.
            box_maxes = box[3:6] / 2.
            box_mins = -box_maxes
            anchor_maxes = anchor / 2.
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_whd = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_whd[0] * intersect_whd[1] * intersect_whd[2]
            box_area = box[3] * box[4] * box[5]
            anchor_area = anchor[0] * anchor[1] * anchor[2]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = anchor_idx

        if best_iou > 0:
            detectors_mask[i, j, k, best_anchor] = 1
            adjusted_box = np.array(
                [
                    box[0] - j, box[1] - i, box[2] - k,
                    np.log(box[3] / anchors[best_anchor][0]),
                    np.log(box[4] / anchors[best_anchor][1]),
                    np.log(box[5] / anchors[best_anchor][2]), box_class
                ],
                dtype=np.float32)
            matching_true_boxes[i, j, k, best_anchor] = adjusted_box
    return detectors_mask, matching_true_boxes


def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):  # TODO: ** Check this
    """Determine space_to_depth output shape for block_size=2.

    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, input_shape[3] // 2, 4 *
            input_shape[4]) if input_shape[1] else (input_shape[0], None, None, None,
                                                    4 * input_shape[4])


def train(model, classes, anchors, images, boxes, output_path=''):
    '''Train YOLO model.

    Positional arguments:
        model
        classes
        anchors
        images
        boxes

    Optional arguments:
        output_path

    Returns:
        None    there are no explicit outputs for training; the trained models are saved to file
                logs training with tensorboard
                saves training weights in current directory
                best weights according to val_loss is saved as trained_stage_3_best.h5
    '''

    # Create output directory, if needed
    # if output_path != '':
    #     os.makedirs(output_path, exist_ok=True)

    # This is a hack to use the custom loss function in the last layer.
    model.compile(optimizer=Adam(lr=0.01), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    logging = TensorBoard()
    checkpoint = ModelCheckpoint(
        output_path + "_best.h5",
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=15,
        verbose=1,
        mode='auto')

    # Convert ground truth boxes to expected network output
    image_size = images.shape[1:-1]
    detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors, image_size)

    # Train model
    model.fit(
        [images, boxes, detectors_mask, matching_true_boxes],
        np.zeros(len(images)),
        validation_split=0.1,
        batch_size=2,
        epochs=100,
        callbacks=[logging, checkpoint])
    model.save_weights(output_path + ".h5")

    # model.fit(
    #     [images, boxes, detectors_mask, matching_true_boxes],
    #     np.zeros(len(images)),
    #     validation_split=0,
    #     batch_size=2,
    #     epochs=60000,
    #     callbacks=[logging])
    # model.save_weights(output_path + "_stage_2.h5")

    # model.fit(
    #     [images, boxes, detectors_mask, matching_true_boxes],
    #     np.zeros(len(images)),
    #     validation_split=0,
    #     batch_size=2,
    #     epochs=60000,
    #     callbacks=[logging, checkpoint, early_stopping])
    # model.save_weights(output_path + "_stage_3.h5")


def yolo_body(inputs, box_size, num_anchors, num_classes):
    """Create main body of YOLO model.

    Positional arguments:
        inputs          keras Input layer of images; should have shape (?, h, w, d, c)
        box_size        bounding box shape; type=int
        num_anchors     number of anchors; type=int
        num_classes     number of classes; type=int

    Returns:

    """
    darknet = Model(inputs, darknet_body_3d()(inputs))
    conv20 = utils.compose(
        DarknetConv3D_BN_Leaky(512, (3, 3, 3)),
        DarknetConv3D_BN_Leaky(512, (3, 3, 3)))(darknet.output)

    conv13 = darknet.layers[43].output
    conv21 = DarknetConv3D_BN_Leaky(32, (1, 1, 1))(conv13)
    # TODO: Allow Keras Lambda to use func arguments for output_shape?
    # conv21_reshaped = Lambda(
    #    space_to_depth_x2,
    #    output_shape=space_to_depth_x2_output_shape,
    #    name='space_to_depth')(conv21)
    conv21_reshaped = Conv3D(128, (2, 2, 2), strides=2, padding='same')(conv21)

    x = concatenate([conv21_reshaped, conv20])
    x = DarknetConv3D_BN_Leaky(512, (3, 3, 3))(x)
    x = DarknetConv3D(num_anchors * (num_classes + box_size), (1, 1, 1))(x)
    return Model(inputs, x)


def yolo_head(feats, anchors, num_classes):
    """Convert final layer features to bounding box parameters.

    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    num_classes : int
        Number of target classes.

    Returns
    -------
    box_xyz : tensor
        x, y, z box predictions adjusted by spatial location in conv layer.
    box_whd : tensor
        w, h, d box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, depth, num_anchors, box_params.
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, 1, num_anchors, 3])

    # Static implementation for fixed models.
    # TODO: Remove or add option for static implementation.
    # _, conv_height, conv_width, _ = K.int_shape(feats)
    # conv_dims = K.variable([conv_width, conv_height])

    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:4]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_depth_index = K.arange(0, stop=conv_dims[2])

    conv_height_index = K.tile(conv_height_index, [conv_dims[1] * conv_dims[2]])
    conv_width_index = K.expand_dims(conv_width_index, axis=0)
    conv_width_index = K.expand_dims(conv_width_index, axis=0)
    conv_width_index = K.tile(conv_width_index, [conv_dims[0], 1, conv_dims[2]])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_depth_index = K.expand_dims(conv_depth_index, axis=0)
    conv_depth_index = K.expand_dims(conv_depth_index, axis=0)
    conv_depth_index = K.tile(conv_depth_index, [conv_dims[2], conv_dims[0], 1])
    conv_depth_index = K.flatten(K.transpose(conv_depth_index))

    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index, conv_depth_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], conv_dims[2], 1, 3])
    conv_index = K.cast(conv_index, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, conv_dims[0], conv_dims[1], conv_dims[2], num_anchors, num_classes + 7])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 1, 3]), K.dtype(feats))

    # Static generation of conv_index:
    # conv_index = np.array([_ for _ in np.ndindex(conv_width, conv_height)])
    # conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering.
    # conv_index = K.variable(
    #     conv_index.reshape(1, conv_height, conv_width, 1, 2))
    # feats = Reshape(
    #     (conv_dims[0], conv_dims[1], num_anchors, num_classes + 5))(feats)

    # Adjust predictions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
    box_xyz = (K.sigmoid(feats[..., :3]) + conv_index) / conv_dims
    box_whd = K.exp(feats[..., 3:6]) * anchors_tensor / conv_dims
    box_confidence = K.sigmoid(feats[..., 6:7])
    box_class_probs = K.softmax(feats[..., 7:])

    return box_xyz, box_whd, box_confidence, box_class_probs


def yolo_boxes_to_corners(box_xyz, box_whd):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xyz - (box_whd / 2.)
    box_maxes = box_xyz + (box_whd / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_mins[..., 2:3],  # z_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1],  # x_max
        box_maxes[..., 2:3]  # z_max
    ])


def yolo_loss(args, anchors, num_classes, rescore_confidence=False, print_loss=False):
    """YOLO localization loss function.

    Parameters
    ----------
    yolo_output : tensor
        Final convolutional layer features.

    true_boxes : tensor
        Ground truth boxes tensor with shape [batch, num_true_boxes, 7]
        containing box x_center, y_center, z_center, width, height, depth, and class.

    detectors_mask : array
        0/1 mask for detector positions where there is a matching ground truth.

    matching_true_boxes : array
        Corresponding ground truth boxes for positive detector positions.
        Already adjusted for conv height and width.

    anchors : tensor
        Anchor boxes for model.

    num_classes : int
        Number of object classes.

    rescore_confidence : bool, default=False
        If true then set confidence target to IOU of best predicted box with
        the closest matching ground truth box.

    print_loss : bool, default=False
        If True then use a tf.Print() to print the loss components.

    Returns
    -------
    mean_loss : float
        mean localization loss across minibatch
    """
    (yolo_output, true_boxes, detectors_mask, matching_true_boxes) = args
    num_anchors = len(anchors)
    object_scale = 5
    no_object_scale = 1
    class_scale = 1
    coordinates_scale = 1
    pred_xyz, pred_whd, pred_confidence, pred_class_prob = yolo_head(
        yolo_output, anchors, num_classes)

    # Unadjusted box predictions for loss
    # TODO: Remove extra computation shared with yolo_head
    yolo_output_shape = K.shape(yolo_output)
    feats = K.reshape(yolo_output, [
        -1, yolo_output_shape[1], yolo_output_shape[2], yolo_output_shape[3], num_anchors,
        num_classes + 7
    ])
    pred_boxes = K.concatenate(
        (K.sigmoid(feats[..., 0:3]), feats[..., 3:6]), axis=-1)

    # TODO: Adjust predictions by image width/height for non-square images?
    # IOUs may be off due to different aspect ratio

    # Expand pred x,y,z,w,h,d to allow comparison with ground truth
    # batch, conv_height, conv_width, conv_depth, num_anchors, num_true_boxes, box_params
    pred_xyz = K.expand_dims(pred_xyz, 5)
    pred_whd = K.expand_dims(pred_whd, 5)

    pred_whd_half = pred_whd / 2.
    pred_mins = pred_xyz - pred_whd_half
    pred_maxes = pred_xyz + pred_whd_half

    true_boxes_shape = K.shape(true_boxes)

    # batch, conv_height, conv_width, conv_depth, num_anchors, num_true_boxes, box_params
    true_boxes = K.reshape(true_boxes, [
        true_boxes_shape[0], 1, 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
    ])
    true_xyz = true_boxes[..., 0:3]
    true_whd = true_boxes[..., 3:6]

    # Find IOU of each predicted box with each ground truth box
    true_whd_half = true_whd / 2.
    true_mins = true_xyz - true_whd_half
    true_maxes = true_xyz + true_whd_half

    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_whd = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_whd[..., 0] * intersect_whd[..., 1] * intersect_whd[..., 2]

    pred_areas = pred_whd[..., 0] * pred_whd[..., 1] * pred_whd[..., 2]
    true_areas = true_whd[..., 0] * true_whd[..., 1] * true_whd[..., 2]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    # Best IOUs for each location.
    best_ious = K.max(iou_scores, axis=5)
    best_ious = K.expand_dims(best_ious)

    # A detector has found an object if IOU > thresh for some true box
    object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))

    # TODO: Darknet region training includes extra coordinate loss for early
    # training steps to encourage predictions to match anchor priors

    # Determine confidence weights from object and no_object weights
    # NOTE: YOLO does not use binary cross-entropy here
    no_object_weights = (no_object_scale * (1 - object_detections) *
                         (1 - detectors_mask))
    no_objects_loss = no_object_weights * K.square(-pred_confidence)

    if rescore_confidence:
        objects_loss = (object_scale * detectors_mask *
                        K.square(best_ious - pred_confidence))
    else:
        objects_loss = (object_scale * detectors_mask *
                        K.square(1 - pred_confidence))
    confidence_loss = objects_loss + no_objects_loss

    # Classification loss for matching detections
    # NOTE: YOLO does not use categorical cross-entropy loss here
    matching_classes = K.cast(matching_true_boxes[..., 6], 'int32')
    matching_classes = K.one_hot(matching_classes, num_classes)
    classification_loss = (class_scale * detectors_mask *
                           K.square(matching_classes - pred_class_prob))

    # Coordinate loss for matching detection boxes
    matching_boxes = matching_true_boxes[..., 0:6]
    coordinates_loss = (coordinates_scale * detectors_mask *
                        K.square(matching_boxes - pred_boxes))

    confidence_loss_sum = K.sum(confidence_loss)
    classification_loss_sum = K.sum(classification_loss)
    coordinates_loss_sum = K.sum(coordinates_loss)
    total_loss = 0.5 * (
        confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)
    if print_loss:
        total_loss = tf.Print(
            total_loss, [
                total_loss, confidence_loss_sum, classification_loss_sum,
                coordinates_loss_sum
            ],
            message='yolo_loss, conf_loss, class_loss, box_coord_loss:')

    return total_loss


def yolo(inputs, anchors, num_classes):
    """Generate a complete YOLO_v2 localization model."""
    num_anchors = len(anchors)
    body = yolo_body(inputs, num_anchors, num_classes)
    outputs = yolo_head(body.output, anchors, num_classes)
    return outputs


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filter YOLO boxes based on object and class confidence."""
    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    prediction_mask = box_class_scores >= threshold

    # TODO: Expose tf.boolean_mask to Keras backend?
    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)
    return boxes, scores, classes


def yolo_eval(yolo_outputs, image_shape, max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """Evaluate YOLO model on given input batch and return filtered boxes."""
    box_xyz, box_whd, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xyz, box_whd)
    boxes, scores, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=score_threshold)

    # Scale boxes back to original image shape.
    height = image_shape[0]
    width = image_shape[1]
    depth = image_shape[2]
    image_dims = K.stack([height, width, depth, height, width, depth])
    image_dims = K.reshape(image_dims, [1, 6])
    boxes = boxes * image_dims

    # TODO: Something must be done about this ugly hack!
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(
        boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)
    return boxes, scores, classes
