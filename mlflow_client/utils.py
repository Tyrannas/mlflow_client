import os
import inspect


def get_caller_dir_path():
    # from stackoverflow: https://stackoverflow.com/a/55469882/4541360
    # get the caller's stack frame and extract its file path
    frame_info = inspect.stack()[2]
    file_path = frame_info[1]  # in python 3.5+, you can use frame_info.filename
    del frame_info  # drop the reference to the stack frame to avoid reference cycles

    # make the path absolute (optional)
    dir_path = os.path.dirname(os.path.abspath(file_path))
    return dir_path
