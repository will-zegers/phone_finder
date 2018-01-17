import sys
import tensorflow as tf

import init
import util


def main():
    image_path = sys.argv[1]
    GRAPH_PATH = 'phone_detection_graph/frozen_inference_graph.pb'

    detection_graph = util.load_detection_graph(GRAPH_PATH)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            print(util.detect_phone_center(image_path, detection_graph, sess))


if __name__ == '__main__':
    main()
