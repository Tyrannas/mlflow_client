import json
import logging
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Union

from mlflow_client.hooks import with_hooks, Hook
from mlflow_client.utils import log_environnment

try:
    import mlflow
except ImportError:
    pass


class MLFramework(Enum):
    """
    Enum to choose between ml frameworks
    """
    PYFUNC = 0
    SCIKIT_LEARN = 1
    KERAS = 2
    PYTORCH = 3

    def __eq__(self, other):
        if self.value == other.value:
            return True
        else:
            return False


class AbstractRun(ABC):
    """
    Abstract class to defined the base methods of a Run Object
    A Run Object is created
    """

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
    def log_model(self, model, output_dir: str, library: MLFramework = MLFramework.PYFUNC, load_entry_point=None):
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


class LocalRun(AbstractRun):
    """
    Store metrics, parameters and artifacts without any need of importing mlflow
    """
    def __init__(self, path='.', experiment_name="DefaultExperiment", hooks=None):
        # if no path is defined, the run will be created at the location of the file calling LocalBackend
        self._hooks = hooks
        self._run_id = None
        self._run_started = False
        self._experiment_name = experiment_name
        self._path = os.path.join(path, 'mlruns', self._experiment_name)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.end_run()

    def log_metric(self, metric: str, value: Union[int, float]):
        if isinstance(value, str):
            raise ValueError('Metrics should be only numerical')

        path = os.path.join(self._path, 'metrics.json')
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

    def log_parameter(self, parameter: str, value):
        path = os.path.join(self._path, 'parameters.json')
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

    def log_artifact(self, path_to_file, path_to_save = None):
        name = os.path.split(path_to_file)[-1]
        if path_to_save:
            name = os.path.join(path_to_save, name)

        path = os.path.join(self._path, 'artifacts', name)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # TODO: check if log_artifact cannot directly be inputed a buffer instead of a source path
        shutil.copy(path_to_file, path)

    def log_model(self, model, output_dir: str = 'model', library: MLFramework = MLFramework.PYFUNC, load_entry_point=None):

        path = os.path.join(self._path, 'artifacts', output_dir)
        os.makedirs(path, exist_ok=True)

        metadata = self._get_metadata()

        # persist model differently according to used library
        if library == MLFramework.SCIKIT_LEARN:
            import sklearn as sk
            import pickle

            path = os.path.join(path, "model.pkl")
            metadata['scikit-learn'] = sk.__version__

            with open(path, 'wb') as f:
                pickle.dump(model, f)

        elif library == MLFramework.PYFUNC:
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

    @with_hooks(Hook.RUN_STARTED)
    def start_run(self, run_id: int = None):

        self._run_id = run_id or uuid.uuid1()
        self._run_started = True
        self._path = os.path.join(self._path,  str(self._run_id))

        logging.getLogger('mlflow_client').warning(f'Run started with pid {self._run_id} \nLogging at: {self._path}')
        log_environnment(self._path)

    @with_hooks(Hook.RUN_ENDED)
    def end_run(self):
        self._run_started = False

    def _get_metadata(self):
        with open(os.path.join(self._path, "meta.json"), 'r') as f:
            metadata = json.load(f)
        return metadata

    def _save_metadata(self, new_metadata):
        with open(os.path.join(self._path, "meta.json"), 'w') as f:
            json.dump(new_metadata, f, indent=4)


class MLFlowRun(AbstractRun):
    """
    Simple Backend wrapper for the classic mlflow behaviour
    """
    def __init__(self, uri=None, hooks=None, experiment_name=None):
        if uri:
            mlflow.set_tracking_uri(uri)
        self._run_started = False
        self._hooks = hooks

        # set requested experiment if existing else create it
        # FIXME: setting manually experiment might lead into conflicts
        # if experiment_name:
        #     experiment = mlflow.get_experiment_by_name(experiment_name)
        #     if experiment:
        #         mlflow.set_experiment(experiment_name)
        #     else:
        #         mlflow.create_experiment(experiment_name)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.end_run()

    @property
    def _run_id(self):
        return mlflow.active_run().info.run_id

    @property
    def _experiment_id(self):
        return mlflow.active_run().info.experiment_id

    @property
    def _experiment_name(self):
        return mlflow.get_experiment(self._experiment_id).name

    def log_metric(self, metric: str, value: Union[int, float]):
        mlflow.log_metric(metric, value)

    def log_parameter(self, parameter: str, value):
        mlflow.log_param(parameter, value)

    def log_artifact(self, path_to_file, path_to_save=None):
        mlflow.log_artifact(path_to_file, path_to_save)

    def log_model(self, model, output_dir: str = "model", library: MLFramework = MLFramework.PYFUNC, load_entry_point=None):

        if library == MLFramework.SCIKIT_LEARN:
            import mlflow.sklearn
            mlflow.sklearn.log_model(model, output_dir)

        elif library == MLFramework.PYFUNC:
            if not isinstance(model, str):
                raise ValueError("For Pyfunc models, model needs to be a path to persisted model or a path to a "
                                 "directory containing the persisted model(s)")

            import mlflow.pyfunc
            mlflow.pyfunc.log_model(output_dir, loader_module=load_entry_point, data_path=model)
        else:
            pass
            # TODO: implement more classes

    @with_hooks(Hook.RUN_STARTED)
    def start_run(self, run_id: int = None):
        self._run_started = True
        mlflow.start_run(run_id=run_id)

    @with_hooks(Hook.RUN_ENDED)
    def end_run(self):
        mlflow.end_run()