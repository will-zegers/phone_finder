import os
import sys
import platform

# Make sure object_detection and slim libraries are in the Python path
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/slim')

# Compile the protos in object_detection for use in Python TF
if not os.path.exists('./object_detection/protos/eval_pb2.py'):
    print('Compiling proto files...')
    if platform.system() == 'Linux':
        if platform.architecture()[0] == '64bit':
                os.system('./protoc/bin/protoc object_detection/protos/*.proto --python_out=.')