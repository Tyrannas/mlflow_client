import os

import builtins
from functools import partial


class FakeMLFlow:
    @staticmethod
    def set_tracking_uri(uri):
        os.environ['URI'] = uri


def mock_import(name, *args, installed=False):
    """
    Mock mlflow import, installed mocks behaviour if mlflow is supposed to be installed or not
    """
    if name == 'mlflow':
        if not installed:
            raise ImportError
        else:
            return FakeMLFlow()
    else:
        return builtins.__import__(name, *args)


mock_mlflow_not_installed = partial(mock_import, installed=False)


mock_mlflow_installed = partial(mock_import, installed=True)
