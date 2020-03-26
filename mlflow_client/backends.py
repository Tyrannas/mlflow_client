from __future__ import annotations

import uuid
import os
import sys
import json
import logging
import shutil

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from contextlib import contextmanager
from typing import Iterable, Union

from mlflow_client.utils import get_caller_dir_path


class MLFrameworks(Enum):
    """
    Enum to choose between ml frameworks
    """
    PYFUNC = 0
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

    infos['requirements'] = list(freeze.freeze())
    infos['python'] = sys.version.split(' ')[0]

    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, 'meta.json'), 'w') as f:
        json.dump(infos, f, indent=4)


def if_run_active(method):
    """
    Prevent logging if no active run is found in the current backend
    """
    def wrapper(self, *args, **kwargs):
        if self._run_started:
            return method(self, *args, **kwargs)
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
            logging.getLogger("mlflow_client").warning(f"AutoBackend configured with MLFlow Distant Backend and URI: {uri}")
            return MLFlowBackend(uri)
        else:
            logging.getLogger("mlflow_client").warning(f"AutoBackend configured with MLFlow Local Backend")
            return MLFlowBackend()
    except ImportError:
        logging.getLogger("mlflow_client").warning(f"AutoBackend configured with LocalStorage Backend")
        return LocalBackend()


class Backend(ABC):
    @abstractmethod
    def log_metric(self, metric: str, value: Union[int, float]):
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
    def log_model(self, model, output_dir: str, library: MLFrameworks = MLFrameworks.PYFUNC, load_entry_point=None):
        """
        Save the resulting model of the run
        :param model: an instance of a class, a persisted instance, or a directory
        :param output_dir: the name of the dir where the model infos and the model will be persisted
        :param library: the framework used to train the model, must be one of @MLFrameworks
        :param load_entry_point:
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
        # if no path is defined, the run will be created at the location of the file calling LocalBackend
        root_path = path or get_caller_dir_path()
        self._path = os.path.join(root_path, 'mlruns')
        self._run_started = 0

    @if_run_active
    def log_metric(self, metric: str, value: Union[int, float]):
        if isinstance(value, str):
            raise ValueError('Metrics should be only numerical')

        path = os.path.join(self._path, str(self._run_started), 'metrics.json')
        # load existing metrics
        if os.path.exists(path):
            with open(path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {}

        # create entry if not existing and add value to it
        metrics.setdefault(metric, []).append(value)

        # write new metrics
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)

    @if_run_active
    def log_parameter(self, parameter: str, value):
        path = os.path.join(self._path, str(self._run_started), 'parameters.json')
        # load existing parameters
        if os.path.exists(path):
            with open(path, 'r') as f:
                parameters = json.load(f)
        else:
            parameters = {}

        # create entry if not existing and add value to it
        parameters.setdefault(parameter, []).append(value)

        # write new parameters
        with open(path, 'w') as f:
            json.dump(parameters, f, indent=4)

    @if_run_active
    def log_artifact(self, path_to_file, path_to_save = None):
        name = os.path.split(path_to_file)[-1]
        if path_to_save:
            name = os.path.join(path_to_save, name)

        path = os.path.join(self._path, str(self._run_started), 'artifacts', name)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # TODO: check if log_artifact cannot directly be inputed a buffer instead of a source path
        shutil.copy(path_to_file, path)

    @if_run_active
    def log_model(self, model, output_dir: str = 'model', library: MLFrameworks = MLFrameworks.PYFUNC, load_entry_point=None):

        path = os.path.join(self._path, str(self._run_started), 'artifacts', output_dir)
        os.makedirs(path, exist_ok=True)

        metadata = self._get_metadata()

        # persist model differently according to used library
        if library == MLFrameworks.SCIKIT_LEARN:
            import sklearn as sk
            import pickle

            path = os.path.join(path, "model.pkl")
            metadata['scikit-learn'] = sk.__version__

            with open(path, 'wb') as f:
                pickle.dump(model, f)

        elif library == MLFrameworks.PYFUNC:
            if not isinstance(model, str):
                raise ValueError("For Pyfunc models, model needs to be a path to persisted model or a path to a "
                                 "directory containing the persisted model(s)")
            if not load_entry_point:
                raise ValueError("For Pyfunc models, param load_entry_point needs to be passed")

            metadata['loadEntryPoint'] = load_entry_point

            # respect mlflow directories
            path = os.path.join(path, 'data')

            # in case of py_func, model is a path or a directory and needs to be copied to the output directory
            if os.path.isdir(model):
                path = os.path.join(path, os.path.split(model)[-1])
                shutil.copytree(model, path)
            else:
                shutil.copy(model, path)
        else:
            pass
            # TODO: support more libraries

        metadata['updateTime'] = str(datetime.now())
        self._save_metadata(metadata)

    @contextmanager
    def start_run(self, run_id: int = None) -> Iterable[LocalBackend]:

        self._run_started = run_id or uuid.uuid1()
        if self._run_started == 0:
            raise ValueError("Run Id cannot be None or 0")

        log_path = os.path.join(self._path, str(self._run_started))
        logging.getLogger('mlflow_client').warning(f'Run started with pid {self._run_started} \nLogging at: {log_path}')
        log_environnment(log_path)

        try:
            yield self
        finally:
            self.end_run()

    @if_run_active
    def end_run(self):
        self._run_started = 0

    def _get_metadata(self):
        with open(os.path.join(self._path, str(self._run_started), "meta.json"), 'r') as f:
            metadata = json.load(f)
        return metadata

    def _save_metadata(self, new_metadata):
        with open(os.path.join(self._path, str(self._run_started), "meta.json"), 'w') as f:
            json.dump(new_metadata, f, indent=4)


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
    def log_metric(self, metric: str, value: Union[int, float]):
        self._mlflow.log_metric(metric, value)

    @if_run_active
    def log_parameter(self, parameter: str, value):
        self._mlflow.log_param(parameter, value)

    @if_run_active
    def log_artifact(self, path_to_file, path_to_save=None):
        self._mlflow.log_artifact(path_to_file, path_to_save)

    @if_run_active
    def log_model(self, model, output_dir: str = "model", library: MLFrameworks = MLFrameworks.PYFUNC, load_entry_point=None):

        if library == MLFrameworks.SCIKIT_LEARN:
            import mlflow.sklearn
            mlflow.sklearn.log_model(model, output_dir)

        elif library == MLFrameworks.PYFUNC:
            if not isinstance(model, str):
                raise ValueError("For Pyfunc models, model needs to be a path to persisted model or a path to a "
                                 "directory containing the persisted model(s)")
            # model_path = model if os.path.isdir(model) else os.path.join(get_caller_dir_path(), model)
            import mlflow.pyfunc
            mlflow.pyfunc.log_model(output_dir, loader_module=load_entry_point, data_path=model)
        else:
            pass
            # TODO: implement more classes

    @contextmanager
    def start_run(self, run_id: int = None) -> Iterable[MLFlowBackend]:
        self._run_started = True
        self._mlflow.start_run()
        try:
            yield self
        finally:
            self.end_run()

    @if_run_active
    def end_run(self):
        self._mlflow.end_run()





