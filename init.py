import os
import sys
import platform

# Make sure object_detection and slim libraries are in the Python path
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/slim')

# Compile the protos in object_detection for use in Python TF
if not os.path.exists('./object_detection/protos/eval_pb2.py'):
    print('Compiling proto files...')
    _platform = platform.system()
    if _platform == 'Darwin':
        protoc = 'protoc_osx_x86_64'
    elif _platform == 'Linux' or _platform == 'Linux2':
        if platform.architecture()[0] == '32bit':
            protoc = 'protoc_linux_x86_32'
        else:
            protoc = 'protoc_linux_x86_64'
    else:
        print('Operating system not supported')
        exit(0)

	
    os.system('./protoc/bin/{} object_detection/protos/*.proto --python_out=.'.format(protoc))
