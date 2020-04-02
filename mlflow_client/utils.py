import json
import os
import inspect
import sys


def get_caller_dir_path():
    # from stackoverflow: https://stackoverflow.com/a/55469882/4541360
    # get the caller's stack frame and extract its file path
    frame_info = inspect.stack()[2]
    file_path = frame_info[1]  # in python 3.5+, you can use frame_info.filename
    del frame_info  # drop the reference to the stack frame to avoid reference cycles

    # make the path absolute (optional)
    dir_path = os.path.dirname(os.path.abspath(file_path))
    return dir_path


def log_environnment(path):
    """
    Utility function to persist the whole python environnment
    :return:
    """
    infos = {}

    # pip freeze
    try:
        from pip._internal.operations import freeze
    except ImportError:  # pip < 10.0
        from pip.operations import freeze

    infos['requirements'] = list(freeze.freeze())
    infos['python'] = sys.version.split(' ')[0]

    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, 'meta.json'), 'w') as f:
        json.dump(infos, f, indent=4)