import uuid
import os
import sys
import json
import logging

from abc import ABC, abstractmethod
from enum import Enum
from contextlib import contextmanager


class MLFrameworks(Enum):
    """
    Enum to choose between ml frameworks
    """
    PYTHON = 0
    SCIKIT_LEARN = 1
    KERAS = 2
    PYTORCH = 3


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

    infos['requirements'] = freeze.freeze()
    infos['python'] = sys.version.split(' ')[0]

    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, 'meta.json'), 'w') as f:
        json.dump(infos, f)


def if_run_active(method):
    """
    Prevent logging if no active run is found in the current backend
    """
    def wrapper(self, *args):
        if self._run_started:
            return method(self, *args)
        else:
            raise ValueError("Logging method called without any active run")
    return wrapper


def get_auto_backend():
    """
    Returns a backend automatically
    LocalBackend if mlflow is not found
    MLFLow Local if mlflow is found
    MLFlow Server if mlflow is found and MLFLOW_TRACKING_URI defined in env variables
    """
    try:
        import mlflow
        if 'MLFLOW_TRACKING_URI' in os.environ:
            uri = os.environ['MLFLOW_TRACKING_URI']
            logging.getLogger("mlflow_client").info(f"AutoBackend configured with MLFlow Distant Backend and URI: {uri}")
            return MLFlowBackend(uri)
        else:
            logging.getLogger("mlflow_client").info(f"AutoBackend configured with MLFlow Local Backend")
            return MLFlowBackend()
    except ImportError:
        logging.getLogger("mlflow_client").info(f"AutoBackend configured with LocalStorage Backend")
        return LocalBackend()


class Backend(ABC):
    @abstractmethod
    def log_metric(self, metric: str, value: int):
        """
        Log a metric (only numeric)
        """
        pass

    @abstractmethod
    def log_parameter(self, parameter: str, value):
        """
        Log a parameter that was used to launch the train, value can be any type
        """
        pass

    @abstractmethod
    def log_artifact(self, path_to_file: str, path_to_save:str = None):
        """
        Save a file, can be text, image, bytes ...
        :param path_to_file: the path to the local file that needs to be saved as an artifact
        :param path_to_save: optional path, if None: the artifact will be stored under /artifacts
        if specified it will be stored under /artifacts/path_to_save
        """
        pass

    @abstractmethod
    def log_model(self,  model, name: str, library: MLFrameworks = MLFrameworks.PYTHON):
        """
        Save the resulting model of the run
        :param model: an instance of a class
        :param name: the name of the model
        :param library: the framework used to train the model, must be one of @MLFrameworks
        :return:
        """
        pass

    @abstractmethod
    def start_run(self, run_id: int = None):
        """
        Start a new run
        :param run_id: optional run id parameter, if filled, will be used to log into an existing run
        """
        pass

    @abstractmethod
    def end_run(self):
        """
        End the current run
        """
        pass


class LocalBackend(Backend):
    """
    Store metrics, parameters and artifacts without any need of importing mlflow
    """
    def __init__(self, path=None):
        root_path = path if path else os.path.dirname(__file__)
        self._path = os.path.join(root_path, 'mlruns')
        self._run_started = 0

    @if_run_active
    def log_metric(self, metric: str, value: int):
        path = os.path.join(self._path, str(self._run_started), 'metrics.json')
        # load existing metrics
        if os.path.exists(path):
            with open(path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {}
        metrics[metric] = value
        # write new metrics
        with open(path, 'w') as f:
            json.dump(metrics, f)

    @if_run_active
    def log_parameter(self, parameter: str, value):
        path = os.path.join(self._path, str(self._run_started), 'parameters.json')
        # load existing parameters
        if os.path.exists(path):
            with open(path, 'r') as f:
                parameters = json.load(f)
        else:
            parameters = {}
        parameters[parameter] = value
        # write new parameters
        with open(path, 'w') as f:
            json.dump(parameters, f)

    @if_run_active
    def log_artifact(self, path_to_file, path_to_save = None):
        name = os.path.split(path_to_file)[-1]
        if path_to_save:
            name = os.path.join(path_to_save, name)

        path = os.path.join(self._path, str(self._run_started), 'artifacts', name)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # TODO: check if log_artifact cannot directly be inputed a buffer instead of a source path
        with open(path_to_file, 'rb') as file:
            with open(path, 'wb') as artifact:
                artifact.write(file.read())

    @if_run_active
    def log_model(self, model, name: str = None, library: MLFrameworks = MLFrameworks.PYTHON):
        if not name:
            name = str(uuid.uuid1()) + '.pkl'

        path = os.path.join(self._path, str(self._run_started), 'artifacts', 'model', name)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if library == MLFrameworks.SCIKIT_LEARN:
            import sklearn as sk
            import pickle

            with open(os.path.join(self._path, str(self._run_started), "meta.json"), 'r') as f:
                metadata = json.load(f)

            metadata['scikit-learn'] = sk.__version__
            with open(path, 'wb') as f:
                pickle.dump(model, f)

        elif library == MLFrameworks.PYTHON:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        else:
            pass
            # TODO: support more libraries

    @contextmanager
    def start_run(self, run_id: int = None):
        self._run_started = run_id if run_id else uuid.uuid1()
        log_environnment(os.path.join(self._path, str(self._run_started)))
        try:
            yield self
        finally:
            self.end_run()

    @if_run_active
    def end_run(self):
        self._run_started = 0


class MLFlowBackend(Backend):
    """
    Simple Backend wrapper for the classic mlflow behaviour
    """
    def __init__(self, uri=None):
        try:
            import mlflow
            self._mlflow = mlflow
        except ImportError:
            raise ImportError("Trying to create an MLFlowBackend while mlflow is not present on the machine")
        if uri:
            mlflow.set_tracking_uri(uri)
        self._run_started = 0

    @if_run_active
    def log_metric(self, metric: str, value: int):
        self._mlflow.log_metric(metric, value)

    @if_run_active
    def log_parameter(self, parameter: str, value):
        self._mlflow.log_param(parameter, value)

    @if_run_active
    def log_artifact(self, path_to_file, path_to_save=None):
        self._mlflow.log_artifact(path_to_file, path_to_save)

    @if_run_active
    def log_model(self, model, name: str = "model", library: MLFrameworks = MLFrameworks.PYTHON):
        if library == MLFrameworks.SCIKIT_LEARN:
            import mlflow.sklearn
            mlflow.sklearn.log_model(model, name)
        else:
            pass
            # TODO: implement more classes, Note: pyfunc seem to be more complicated

    @contextmanager
    def start_run(self, run_id: int = None):
        self._run_started = True
        try:
            yield self._mlflow.start_run()
        finally:
            self.end_run()

    @if_run_active
    def end_run(self):
        self._mlflow.end_run()





