import os
import logging

from enum import Enum

from hooks import resolve_hooks, Hook
from runs import MLFlowRun, LocalRun


class Backend(Enum):
    LOCAL = 0
    MLFLOW = 1


def get_auto_run(hooks):
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
            return MLFlowRun(uri, hooks=hooks)
        else:
            logging.getLogger("mlflow_client").warning(f"AutoBackend configured with MLFlow Local Backend")
            return MLFlowRun(hooks=hooks)
    except ImportError:
        logging.getLogger("mlflow_client").warning(f"AutoBackend configured with LocalStorage Backend")
        return LocalRun(hooks=hooks)


class MLClient:
    def __init__(self, backend: Backend = None, backend_uri=None, experiment="default_experiment", hooks_uri=None):
        self._hooks = resolve_hooks(hooks_uri)
        self._backend = backend
        self._experiment = experiment
        self._backend_uri = backend_uri
        self._run = None

    def start_run(self, run_id=None):
        if not self._backend:
            self._run = get_auto_run(self._hooks)

        elif self._backend == Backend.LOCAL or isinstance(self._backend, str) and self._backend.lower() == "local":
            backend_uri = self._backend_uri or '.'
            self._run = LocalRun(path=backend_uri, experiment_name=self._experiment, hooks=self._hooks)

        elif self._backend == Backend.MLFLOW or isinstance(self._backend, str) and self._backend.lower() == "mlflow":
            try:
                import mlflow
            except ImportError:
                raise ImportError("Trying to create an MLFlowBackend while mlflow is not present on the machine")

            self._run = MLFlowRun(uri=self._backend_uri, experiment_name=self._experiment, hooks=self._hooks)
        else:
            raise ValueError(f"Unrecognized type of backend requested: {self._backend}")

        self._run.start_run(run_id)
        return self._run

    def end_run(self):
        self._run.end_run()
        self._run = None

    def add_hook(self, event: Hook, url: str, name: str = None):
        if not name:
            name = event.name.lower() + '-hook'
        self._hooks.setdefault(event.name.lower(), []).append(dict(url=url, name=name))




