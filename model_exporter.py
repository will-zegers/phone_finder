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

from object_detection import exporter
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def export():
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

