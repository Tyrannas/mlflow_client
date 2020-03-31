import datetime
from enum import Enum

import requests

"""
Hooks template file:
{
     'run_started': [
        {
            'name': 'firstHook'
            'url': 'http://sgaiapi/new_run
        }
        ...
     ]
     ...
}
"""


class Hooks(Enum):
    RUN_STARTED = 0
    RUN_ENDED = 1


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
    # TODO: implement behaviour
    return {}


def send_hook(action: Hooks, experiment_name, run_id, url):
    """
    Create the hook object and send it to the specified URI
    """
    hook = {
        'action': action.name,
        'timestamp': str(datetime.datetime.now()),
        'experiment': experiment_name,
        'run': run_id
    }
    requests.post(url, json=hook)