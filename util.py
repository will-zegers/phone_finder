import os
import cv2
import numpy as np
import tensorflow as tf

import init

from object_detection import exporter
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


def export_graph():
    """Export a checkpoint from the training phase into a frozen inference graph .pb file.

    :return: None
    """
    ckpts = set([])
    for f in os.listdir('./data/'):
        if 'model.ckpt-' in f:
            prefix = f[f.find('-')+1:]
            step = prefix[:prefix.find('.')]
            ckpts.add(int(step))
    recent_ckpt = max(ckpts)
    print('Exporting model.ckpt-{}'.format(recent_ckpt))

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile('data/pipeline.config', 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    exporter.export_inference_graph('image_tensor', pipeline_config,
                                    'data/model.ckpt-{}'.format(recent_ckpt),
                                    'phone_detection_graph', None)


def load_data(dir_path='./find_phone/'):
    """Loads data and labels from a given directory

    :param dir_path: Location of the directory containing the training examples and labels
    :return: Two arrays, one containing the strings of file name (data), and the other containing
    the corresponding labels the indicate the center XY coordinate of the phone
    """
    data = np.empty(0)
    labels = np.empty((0, 2))
    with open(os.path.join(dir_path, 'labels.txt')) as f:
        line = f.readline()
        while line:
            label = line.split()
            data = np.append(data, os.path.join(dir_path, label[0]))
            labels = np.vstack((labels, (float(label[1]), float(label[2]))))
            line = f.readline()
    return data, labels


def detect_phone_center(image_path, detection_graph, sess):
    """Uses the given detection graph to detect the center location of a phone

    :param image_path: Path to the image on which to detect the phone
    :param detection_graph: A trained TF graph object
    :param sess: The current TF session
    :return: The predicted XY point of the phone center
    """
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image, axis=0)

    (boxes, scores) = sess.run(
        [detection_boxes, detection_scores],
        feed_dict={image_tensor: image_expanded})

    box = boxes[0, 0]
    return (round((box[1] + box[3])/2, 4),
            round((box[0] + box[2])/2, 4))


def load_detection_graph(graph_path):
    """Load a trained graph into memory to be used for predicting

    :param graph_path: Path to the .pb graph file
    :return: A TF Graph object, specified by the given .pb file
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def evaluate(graph_path='phone_detection_graph/frozen_inference_graph.pb',
             image_dir='./find_phone/'):
    """Evaluate the accuracy of the trained object detection graph over several input

    :param graph_path: Path to the .pb graph file
    :param image_dir: Path to the sample inputs and label.txt on which to evaluate the model
    :return: The error rate of the model (i.e., instances when it couldn't detect the phone
    """

    detection_graph = load_detection_graph(graph_path)
    data, labels = load_data(image_dir)

    errors = np.zeros((len(data), 1))
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for i in range(len(data)):
                center = detect_phone_center(data[i], detection_graph, sess)
                if abs(center[0] - labels[i][0]) > 0.05 or abs(center[1] - labels[i][1]) > 0.05:
                    errors[i] = 1
    return errors.sum() / errors.shape[0]
