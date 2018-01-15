#!/bin/nash

source run.sh
python object_detection/train.py --logtostderr --train_dir=data/ --pipeline_config_path=data/pipeline.config
