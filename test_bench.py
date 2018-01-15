import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from object_detection.utils import dataset_util


def load_training_data(dir_path="./find_phone/"):
    data = np.empty(0)
    labels = np.empty((0, 2))
    with open(os.path.join(dir_path, "labels.txt")) as f:
        line = f.readline()
        while line:
            label = line.split()
            data = np.append(data, label[0])
            labels = np.vstack((labels, (float(label[1]), float(label[2]))))
            line = f.readline()
    return data, labels


def create_record_df():

    img_dir = './find_phone/'
    data, labels = load_training_data(img_dir)

    box_size = 0.06
    columns=['height', 'width', 'filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    df = pd.DataFrame(index=[i for i in range(len(data))], columns=columns)
    for i, (filename, center) in enumerate(zip(data, labels)):
        img = cv2.imread(os.path.join(img_dir, filename))
        h, w, _ = img.shape
        pt1 = (round(center[0] - box_size, 4), round(center[1] - box_size, 4))
        pt2 = (round(center[0] + box_size, 4), round(center[1] + box_size, 4))
        df.iloc[i] = np.array([h, w, filename, pt1[0], pt1[1], pt2[0], pt2[1], "phone"])
    return df

def create_label_map(ids, names, filename):
    pbtxt = open('./data/{}.pbtxt'.format(filename), 'w')
    for id, name in zip(ids, names):
        pbtxt.write('{\n')
        pbtxt.write('\tid: {}\n'.format(id))
        pbtxt.write('\tname: {}\n'.format(name))
        pbtxt.write('}\n')

def create_tf_example(img_dir, example):

    with tf.gfile.GFile(os.path.join(img_dir, example['filename']), 'rb') as fid:
        encoded_image_data = fid.read()
    height = int(example['height'])
    width = int(example['width'])
    filename = example['filename']
    image_format = b'jpg'

    xmins = [float(example['xmin'])]
    xmaxs = [float(example['xmax'])]

    ymins = [float(example['ymin'])]
    ymaxs = [float(example['ymax'])]

    classes_text = [example['class']]
    classes = [1]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
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
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter('./data/phone_finder.record')

    img_dir = './find_phone/'
    create_label_map([1], ['Phone'], 'phone_finder')
    examples = create_record_df()

    for i in range(examples.shape[0]):
        example = examples.iloc[i]
        tf_example = create_tf_example(img_dir, example)
        writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == '__main__':
  tf.app.run()
