import os
import cv2
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold

import init
import util

from object_detection import train as od_train
from object_detection.utils import dataset_util


def create_record_df(img_dir, box_size=0.06):
    """
    Compiles a Pandas dataframe containing relevant information of images and the location
    of the phone in each.

    :param img_dir: Location of the directory containing the training examples and labels
    :param box_size: Normalized size of the box in which the image of the phone will be contained
    :return: A Pandas dataframe, where each entry contains the height, width, and filename of an
    image in the training set, plus the normalized points of the box that contains the phone image
    (represented by its top right corner and bottom left)
    """
    data, labels = util.load_data(img_dir)

    columns = ['height', 'width', 'filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    df = pd.DataFrame(index=[i for i in range(len(data))], columns=columns)
    for i, (filename, center) in enumerate(zip(data, labels)):
        img = cv2.imread(filename)
        xmin = round(center[0] - box_size, 4) if center[0] > box_size else 0
        ymin = round(center[1] - box_size, 4) if center[0] > box_size else 0
        xmax = round(center[0] + box_size, 4) if center[0] + box_size <= 1 else 1
        ymax = round(center[1] + box_size, 4) if center[0] + box_size <= 1 else 1
        df.iloc[i] = np.array([img.shape[0], img.shape[1], filename, xmin, ymin, xmax, ymax, 'phone'])
    return df


def create_label_map(names, filename):
    """
    A mapping used by the TensorFlow object detection, associating class names with a corresponding
    id (in this case, we just have a single 'phone' class

    :param names: Class names (phone)
    :param filename: location of where to save the .pbtxt file
    :return: None
    """
    pbtxt = open('./data/{}.pbtxt'.format(filename), 'w')
    for class_id, name in enumerate(names):
        pbtxt.write('item {\n')
        pbtxt.write('\tid: {}\n'.format(class_id+1))
        pbtxt.write('\tname: \"{}\"\n'.format(name))
        pbtxt.write('}\n')


def create_tf_record(img_dir, sample):
    """
    Mostly copy and pasted from https://git.io/vNWx2. This takes in a Pandas dataframe entry and
    converts it to the TFRecord format that can be used by Tensorflow

    :param img_dir: Location of the directory containing the training examples and labels
    :param sample: A single row of a Pandas dataframe, containing all necessary fields needed
    by the TensorFlow object detection API
    :return: An TF record object
    """

    print(sample['filename'])
    with tf.gfile.GFile(sample['filename'], 'rb') as fid:
        encoded_image_data = fid.read()
    height = int(sample['height'])
    width = int(sample['width'])
    filename = sample['filename']
    image_format = b'jpg'

    xmins = [float(sample['xmin'])]
    xmaxs = [float(sample['xmax'])]

    ymins = [float(sample['ymin'])]
    ymaxs = [float(sample['ymax'])]

    classes_text = [sample['class']]
    classes = [1]

    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))


def write_tf_records(img_dir, data, n_splits=5, random_state=451):
    """
    Splits a Pandas dataframe into a training and validation set, and writes it out to a
    TF record file, for use in the TF object detection API

    :param img_dir: Location of the directory containing the training examples and labels
    :param data: A Pandas dataframe containing information pertaining to the training data
    :param n_splits: Number of splits to use in creating the training and validation sets
    :param random_state: Seed for the KFold split
    :return: None
    """
    train_writer = tf.python_io.TFRecordWriter('./data/phone_finder_train.record')
    valid_writer = tf.python_io.TFRecordWriter('./data/phone_finder_valid.record')

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    train_idx, valid_idx = next(kf.split([i for i in range(data.shape[0])]))
    for i in range(data.shape[0]):
        tf_example = create_tf_record(img_dir, data.iloc[i])

        writer = valid_writer if i in valid_idx else train_writer
        writer.write(tf_example.SerializeToString())

    train_writer.close()
    valid_writer.close()


def main(_):
    img_dir = sys.argv[1]
    create_label_map(['Phone'], 'phone_finder')
    data = create_record_df(img_dir)
    write_tf_records(img_dir, data)

    od_train.FLAGS.train_dir = 'data/'
    od_train.FLAGS.pipeline_config_path = 'data/pipeline.config'
    od_train.main(0)

    util.export_graph()


if __name__ == '__main__':
    tf.app.run()
