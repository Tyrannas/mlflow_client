# MLFlow Client
MLFlow wrapper with controllable dependencies
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
#### Code
Just select the backend that you want to use:
- LocalBackend: without MLFLow, reproduce the behaviour
- MLFLowBackend: classic MLFLow

All methods will work whatever backend you're choosing
```python
from mlflow_client import get_auto_backend, LocalBackend, MLFlowBackend, MLFrameworks
from sklearn.dummy import  DummyClassifier

# mlf = LocalBackend() # runs will be stored in current directory
# mlf = LocalBackend('/output_directory_for_runs') # specify path
# mlf = MLFLowBackend() => local MLFLow
mlf = MLFlowBackend(uri='http://some-mlflow-server:5000') # distant MLFLow

# you can use get_auto_backend() to get automatically a backend suiting to your device
# mlf = get_auto_backend()

with mlf.start_run():
    model = DummyClassifier()
    ... train the model
    mlf.log_parameter('alpha', 0.05)
    mlf.log_metric('loss', 3.30)
    mlf.log_artifact('/path/to/some/plot.png')
    # specify the type of the model that you want to persist
    mlf.log_model(model=model, output_dir='dummy_model', library=MLFrameworks.SCIKIT_LEARN)
```

If your model is not pure scikit or not pure (any other ml library), you can persist it as PyFunc:
You just need to specify a path to your persisted model or a path to a directory containing these file(s) and an entry_point allowing to load the files in memory before making a prediction

For example if you created a class MyClass that's doing some preprocessing before fitting, you can initiate this class, do whatever you want, and then persist files as pkl in a directory 'local_path'
You then just need to specify the load_entry_point which is a function / method that takes a path containing the pkl as parameter and returns any object exposing a predict method.

*In the below example this method is the constructor of MyClass*
```python
... 
with mlf.start_run():
    ...
    model = MyClass()
    model.preprocess()
    model.fit()
    model.save_models('local_path')
    # note: library can be omitted since default value is already PYFUNC
    mlf.log_model(model='local_path', load_entry_point='my_module.core.MyClass', library=MLFrameworks.PYFUNC)
```
