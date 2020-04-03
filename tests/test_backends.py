import json
import os
import tempfile
import pickle
import pytest

from unittest.mock import patch

from mlflow_client import MLFramework, MLClient, Backend
from mlflow_client.runs import LocalRun, MLFlowRun
from mlflow_client.client import get_auto_run

from .utils import mock_mlflow_installed, mock_mlflow_not_installed


def create_run():
    """
    Check if every backend can be created without problem
    """
    LocalRun()
    LocalRun('.')
    MLFlowRun()
    MLFlowRun('http://localhost:5000')


# The following tests are all on LocalBackend, one suppose that mlflow unit tests are already written

def test_log_parameter():
    """
    Test parameters logging
    """
    with tempfile.TemporaryDirectory() as path:
        client = MLClient(backend='local', backend_uri=path, experiment="test_exp")
        with client.start_run(1) as run:
            # log a float
            run.log_parameter('alpha', 0.05)
            # log a string
            run.log_parameter('optimizer', 'adam')

            file_path = os.path.join(path, 'mlruns', 'test_exp', '1', 'parameters.json')
            assert os.path.exists(file_path)

            with open(file_path, 'r') as f:
                parameters = json.load(f)
                assert parameters['alpha'][0] == 0.05


def test_log_metric():
    """
    Test metrics logging
    """
    with tempfile.TemporaryDirectory() as path:
        client = MLClient(backend='local', backend_uri=path, experiment="test_exp")
        with client.start_run(1) as run:
            # log a float
            run.log_metric('mse', 3.5)
            # log a string
            with pytest.raises(ValueError):
                run.log_metric('optimizer', 'adam')

            file_path = os.path.join(path, 'mlruns', 'test_exp', '1', 'metrics.json')
            assert os.path.exists(file_path)

            with open(file_path, 'r') as f:
                parameters = json.load(f)
                assert parameters['mse'][0] == 3.5


def test_log_artifact():
    """
    Test artifacts logging
    """
    with tempfile.TemporaryDirectory() as path:
        client = MLClient(backend=Backend.LOCAL, backend_uri=path, experiment="test_exp")
        with client.start_run(1) as run:
            base_path = os.path.join(path, 'mlruns', 'test_exp', '1', 'artifacts')

            # log a text file
            source_path = os.path.join(path, 'test.txt')
            with open(source_path, 'w') as f:
                f.write('this is a test')
            run.log_artifact(source_path)

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
        client = MLClient(backend=Backend.LOCAL, backend_uri=path, experiment="test_exp")
        with client.start_run(1) as run:
            fake_model = dict(predict="fake")
            base_path = os.path.join(path, 'mlruns', 'test_exp', '1', 'artifacts', 'output_dir', 'model.pkl')
            run.log_model(fake_model, 'output_dir', MLFramework.SCIKIT_LEARN)

            assert os.path.exists(base_path)
            with open(base_path, 'rb') as f:
                d = pickle.load(f)
                assert d['predict'] == 'fake'


def test_log_pyfunc_model():
    with tempfile.TemporaryDirectory() as path:
        client = MLClient(backend='local', backend_uri=path, experiment="test_exp")
        with client.start_run(1) as run:
            output_path = os.path.join(path, 'mlruns', 'test_exp', '1', 'artifacts', 'models', 'data')
            fake_model = dict(predict="fake")
            source_path = os.path.join(path, 'resources')
            os.makedirs(source_path)

            with open(os.path.join(source_path, 'model.pkl'), 'wb') as f:
                pickle.dump(fake_model, f)

            # log a whole directory
            run.log_model(source_path, 'models', load_entry_point='my_module.my_class', library=MLFramework.PYFUNC)

            output_file = os.path.join(output_path, 'resources', 'model.pkl')
            assert os.path.exists(output_file)
            assert os.path.isfile(output_file)

            # log a single file
            run.log_model(os.path.join(source_path, 'model.pkl'), 'models', load_entry_point="my_module.my_class", library=MLFramework.PYFUNC)

            output_file = os.path.join(output_path, 'model.pkl')
            assert os.path.exists(output_file)
            assert os.path.isfile(output_file)

            # verify that load_entry_point is mandatory
            with pytest.raises(ValueError):
                run.log_model(source_path, 'models', library=MLFramework.PYFUNC)


def test_auto_backend():
    """
    Test is get_auto_backend returns the appropriate backend depending on the environment
    """
    # if mlflow is not installed
    with patch('builtins.__import__', new=mock_mlflow_not_installed):
        run = get_auto_run()
        assert isinstance(run, LocalRun)

    # if mlflow is installed

    with patch.dict('os.environ', {}, clear=True):
        import mlflow
        # if mlflow is only installed locally
        run = get_auto_run()
        assert isinstance(run, MLFlowRun)

        # if a distant uri is specified
        os.environ['MLFLOW_TRACKING_URI'] = 'mock:5000'
        run = get_auto_run()
        assert mlflow.tracking.get_tracking_uri() == 'mock:5000'
