import json
import os
import tempfile
import pickle
import pytest

from unittest.mock import patch

from mlflow_client import LocalBackend, MLFlowBackend, get_auto_backend, MLFrameworks
from .utils import mock_mlflow_installed, mock_mlflow_not_installed


def create_backends():
    """
    Check if every backend can be created without problem
    """
    LocalBackend()
    LocalBackend('.')
    MLFlowBackend()
    MLFlowBackend('http://localhost:5000')


# The following tests are all on LocalBackend, one suppose that mlflow unit tests are already written


def test_log_parameter():
    """
    Test parameters logging
    """
    with tempfile.TemporaryDirectory() as path:
        mlf = LocalBackend(path)
        with mlf.start_run(1):
            # log a float
            mlf.log_parameter('alpha', 0.05)
            # log a string
            mlf.log_parameter('optimizer', 'adam')

            file_path = os.path.join(path, 'mlruns', '1', 'parameters.json')
            assert os.path.exists(file_path)

            with open(file_path, 'r') as f:
                parameters = json.load(f)
                assert parameters['alpha'] == 0.05


def test_log_metric():
    """
    Test metrics logging
    """
    with tempfile.TemporaryDirectory() as path:
        mlf = LocalBackend(path)
        with mlf.start_run(1):
            # log a float
            mlf.log_metric('mse', 3.5)
            # log a string
            with pytest.raises(ValueError):
                mlf.log_metric('optimizer', 'adam')

            file_path = os.path.join(path, 'mlruns', '1', 'metrics.json')
            assert os.path.exists(file_path)

            with open(file_path, 'r') as f:
                parameters = json.load(f)
                assert parameters['mse'] == 3.5


def test_log_artifact():
    """
    Test artifacts logging
    """
    with tempfile.TemporaryDirectory() as path:
        mlf = LocalBackend(path)
        with mlf.start_run(1):
            base_path = os.path.join(path, 'mlruns', '1', 'artifacts')

            # log a text file
            source_path = os.path.join(path, 'test.txt')
            with open(source_path, 'w') as f:
                f.write('this is a test')
            mlf.log_artifact(source_path)

            # verify that the file exists and is readable
            output_path = os.path.join(base_path, 'test.txt')
            assert os.path.exists(output_path)
            with open(output_path, 'r') as f:
                assert f.read() == 'this is a test'

            # log a binary file
            source_path = os.path.join(base_path, 'test.pkl')
            with open(source_path, 'wb') as f:
                pickle.dump(dict(test="test"), f)

            # verify that the file exists and is readable
            output_path = os.path.join(base_path, 'test.pkl')
            assert os.path.exists(output_path)
            with open(output_path, 'rb') as f:
                dico = pickle.load(f)
                assert dico['test'] == 'test'


def test_log_scikit_model():
    """
    Test scikit-learn model logging
    """
    with tempfile.TemporaryDirectory() as path:
        mlf = LocalBackend(path)
        with mlf.start_run(1):
            fake_model = dict(predict="fake")
            base_path = os.path.join(path, 'mlruns', '1', 'artifacts', 'output_dir', 'model.pkl')
            mlf.log_model(fake_model, 'output_dir', MLFrameworks.SCIKIT_LEARN)

            assert os.path.exists(base_path)
            with open(base_path, 'rb') as f:
                d = pickle.load(f)
                assert d['predict'] == 'fake'


def test_log_pyfunc_model():
    with tempfile.TemporaryDirectory() as path:
        mlf = LocalBackend(path)
        with mlf.start_run(1):
            output_path = os.path.join(path, 'mlruns', '1', 'artifacts', 'models', 'data')
            fake_model = dict(predict="fake")
            source_path = os.path.join(path, 'resources')
            os.makedirs(source_path)

            with open(os.path.join(source_path, 'model.pkl'), 'wb') as f:
                pickle.dump(fake_model, f)

            # log a whole directory
            mlf.log_model(source_path, 'models', load_entry_point='my_module.my_class', library=MLFrameworks.PYFUNC)

            output_file = os.path.join(output_path, 'resources', 'model.pkl')
            assert os.path.exists(output_file)
            assert os.path.isfile(output_file)

            # log a single file
            mlf.log_model(os.path.join(source_path, 'model.pkl'), 'models', load_entry_point="my_module.my_class", library=MLFrameworks.PYFUNC)

            output_file = os.path.join(output_path, 'model.pkl')
            assert os.path.exists(output_file)
            assert os.path.isfile(output_file)

            # verify that load_entry_point is mandatory
            with pytest.raises(ValueError):
                mlf.log_model(source_path, 'models', library=MLFrameworks.PYFUNC)


def test_logging_without_run():
    """
    Test if logging methods are blocked if no run was started before
    """

    mlf = LocalBackend()

    with pytest.raises(ValueError):
        mlf.log_parameter('alpha', 0.05)

    with pytest.raises(ValueError):
        mlf.log_metric('mse', 5)

    with pytest.raises(ValueError):
        mlf.log_artifact('/path/to/file', 'output_dir')

    with pytest.raises(ValueError):
        mlf.log_model('model', 'model')


def test_auto_backend():
    """
    Test is get_auto_backend returns the appropriate backend depending on the environment
    """
    # if mlflow is not installed
    with patch('builtins.__import__', new=mock_mlflow_not_installed):
        mlf = get_auto_backend()
        assert isinstance(mlf, LocalBackend)

    # if mlflow is installed

    with patch.dict('os.environ', {}, clear=True):
        with patch('builtins.__import__', new=mock_mlflow_installed):
            # if mlflow is only installed locally
            mlf = get_auto_backend()
            assert isinstance(mlf, MLFlowBackend)

            # if a distant uri is specified
            os.environ['MLFLOW_TRACKING_URI'] = 'mock:5000'
            mlf = get_auto_backend()
            assert isinstance(mlf, MLFlowBackend)
            assert os.environ.get('URI') == 'mock:5000'
