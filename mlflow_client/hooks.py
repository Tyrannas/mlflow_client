import datetime
import functools
import json
import os
import yaml
import requests

from enum import Enum


"""
Hooks template file:
{
     'run_started': [
        {
            'name': 'firstHook',
            'url': 'http://localhost/hooks
        }
        ...
     ]
     ...
}
"""


class Hook(Enum):
    RUN_STARTED = 0
    RUN_ENDED = 1


def with_hooks(event: Hook):
    """
    Function that takes an event as parameter and returns a decorator that wraps a method,
    execute it and then send a hook if a hook exists.
    In case of failure the status is set to failed
    :param event:
    :return:
    """
    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            error = None
            try:
                method(self, *args, **kwargs)
                status = "success"
                message = None
            except Exception as e:
                status = "failed"
                message = str(e)
                error = e

            if event.name.lower() in self._hooks:
                for hook in self._hooks[event.name.lower()]:
                    send_hook(action=event, experiment_name=self._experiment_name, run_id=self._run_id, url=hook['url'],
                              status=status, message=message)

            # now that the failed hook has been sent, raise the error
            if error:
                raise error

        return wrapper
    return decorator


def resolve_hooks(hooks_uri=None):
    """
    function that try different methods to return hooks
    if hook_uri is set and is an url, it'll try to perform a get on the url and expect to retrieve a json that describe hooks
    if hook_uri is set and is a file uri, it'll try to json load the file
    if hook_uri is set is a folder uri, it'll try to find a MLproject file and parse it
    else it'll try to see if MLFLOW_HOOKS_URI is set
    else it'll return an empty object
    :param hooks_uri:
    :return:
    """
    if not hooks_uri:
        # look into env variables
        if 'MLFLOW_HOOKS_URI' in os.environ:
            hooks_uri = os.environ['MLFLOW_HOOKS_URI']
        else:
            return {}

    # TODO: change this to regex validation? with package validators ?
    try:
        # try to fetch as an url
        res = requests.get(hooks_uri)
        return res.json()

    except requests.exceptions.RequestException:
        if os.path.isdir(hooks_uri):
            # assume there is an MLproject file
            path = os.path.join(hooks_uri, 'MLproject')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    project = yaml.load(f)
                return project['hooks']
            else:
                raise FileNotFoundError("Hooks URI provided is a directory, but no MLproject file found")

        elif os.path.isfile(hooks_uri):
            # assume there is a json containing hooks
            with open(hooks_uri, 'r') as f:
                hooks = json.load(f)
            return hooks

        else:
            raise ValueError("Hooks URI provided not recognized as a valid url, folder or file")


def send_hook(action: Hook, experiment_name, run_id, url, status="success", message=None):
    """
    Create the hook object and send it to the specified URI
    """
    hook = {
        'event': action.name,
        'status': status,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        'payload': {
            'experiment': experiment_name,
            'run': str(run_id),
            'message': message
        }
    }
    requests.post(url, json=hook)