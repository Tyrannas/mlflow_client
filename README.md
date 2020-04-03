# MLFlow Client
MLFlow wrapper with controllable dependencies and support of WebHooks 
Can be used for people who want to experiment locally with mlflow 
# Usage
#### Install
For basic light version use:
```
pip install mlflow-client
```
Else if you want to use the mlflow packaged version:
```
pip install mlflow-client[mlflow]
```
#### Backends
Just create an MLClient
```python
from mlflow_client import MLClient
client = MLClient()
```
MLClient accepts the following optional parameters:
- **backend**: the type of backend that you want to use to store your runs (With or without mlflow). Not specifying the backend will make the Client to provide automatically the backend.
```python
from mlflow_client import Backend
client = MLClient(backend=Backend.MLFLOW) # or MLClient(backend="mlflow")
client = MLClient(backend=Backend.LOCAL) # or MLClient(backend="local")
```
- **backend_uri**: specify where you want to store the results, in case of local storage, backend_uri must be a path to the directory you want to save your results in. In case of an mlflow backend, backend_uri must be an url to a tracking server. 
- **experiment**: the name of your experiment
- **hooks_uri**: this will be detailed in the WebHooks section

#### Runs
Now that your Client is set, you can start a run and with the same syntax as mlflow.
client.start_run() will return a run object that allows you to log parameters, metrics, files and the resulting model.
```python
from sklearn.dummy import  DummyClassifier
from mlflow_client import MLFramework, MLClient, Backend

client = MLClient(backend=Backend.LOCAL)

with client.start_run() as run:
    model = DummyClassifier()
    ... train the model
    run.log_parameter('alpha', 0.05)
    run.log_metric('loss', 3.30)
    run.log_artifact('/path/to/some/plot.png')
    # specify the type of the model that you want to persist
    run.log_model(model=model, output_dir='dummy_model', library=MLFramework.SCIKIT_LEARN)
```

If your model is not pure scikit or not pure (any other ml library), you can persist it as PyFunc:
You just need to specify a path to your persisted model or a path to a directory containing these file(s) and an entry_point allowing to load the files in memory before making a prediction

For example if you created a class MyClass that's doing some preprocessing before fitting, you can initiate this class, do whatever you want, and then persist files as pkl in a directory 'local_path'
You then just need to specify the load_entry_point which is a function / method that takes a path containing the pkl as parameter and returns any object exposing a predict method.

> In the below example this method is the constructor of MyClass
```python
... 
with client.start_run() as run:
    ...
    model = MyClass()
    model.preprocess()
    model.fit()
    model.save_models('local_path')
    # note: library can be omitted since default value is already PYFUNC
    run.log_model(model='local_path', load_entry_point='my_module.core.MyClass', library=MLFrameworks.PYFUNC)
```

You can have a look at all the possibilities [here](https://github.com/Tyrannas/mlflow_client/blob/master/experiments/main.py).

#### WebHooks
MLFlow Client allows you to configure hooks that will be triggered during the run. 
The currently supported format is the following:
```
{
  "event_name" (run_started | run_ended): [
      {
        "url": POST URL TO SEND THE HOOK TO
        "name": HOOK NAME (OPTIONAL)
      }, 
      ...
  ],
  ...
}
```
There are several ways to configure your run to use hooks:
- fill the "hooks_uri" parameter of the MLClient. This uri can be:
    - a GET URL that will return the hooks as JSON
    - a path to a JSON file
    - a path to a directory containing an MLproject, this MLproject needs to have a field hooks (example [here](https://github.com/Tyrannas/mlflow_client/blob/master/experiments/MLproject))
- set the MLFLOW_HOOKS_URI env variable that can be either of the 3 uri above
- use client.add_hook(event=, name=, url=) on an MLClient instance **before** starting the run

When the event occurs the url specified will receive the following payload: 
```
{
    "event": EVENT NAME,
    "status": (success | failed),
    'timestamp': TRIGGER TIME IN FORMAT YYYY-MM-DDTHH:MM:SS,
    'payload': {
        'experiment': EXPERIMENT NAME,
        'run': RUN UUID,
        'message': ADDITIONNAL INFORMATION (CONTAINS THE ERROR LOG IF STATUS FAILED)
    }
}
```
