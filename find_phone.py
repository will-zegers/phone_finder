import os
import cv2
import sys
import platform
import numpy as np
import tensorflow as tf


sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/slim')
if not os.path.exists('./object_detection/protos/eval_pb2.py'):
    print('Compiling proto files...')
    if platform.system() == 'Linux':
        if platform.architecture()[0] == '64bit':
            os.system('./protoc/bin/protoc object_detection/protos/*.proto --python_out=.')

from object_detection.utils import label_map_util


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


GRAPH_NAME = 'frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'phone_finder.pbtxt')

image_path = sys.argv[1]

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(GRAPH_NAME, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image, axis=0)

        (boxes, scores) = sess.run(
            [detection_boxes, detection_scores],
            feed_dict={image_tensor: image_expanded})

box = boxes[0, 0]
center = (round((box[1] + box[3])/2, 4),
          round((box[0] + box[2])/2, 4))
print(center)
