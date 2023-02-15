import os

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def read_path(path):
    return os.path.dirname(path)

def read_filename(path):
    # Read file name excluded file path and extension
    return os.path.basename(path).split('.')[0]

def read_finalfoldername(path):
    return path.split('/')[-1]