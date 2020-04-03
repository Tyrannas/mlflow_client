from mlflow_client import Hook, MLFramework, MLClient, Backend
from sklearn.dummy import DummyClassifier
import numpy as np

if __name__ == "__main__":

    # with an mlflow tracking server
    # backend = MLClient(backend=Backend.MLFLOW, hooks_uri="http://localhost:9999/hooks", backend_uri="http://localhost:5000")

    # with a local client
    # backend = MLClient(backend=Backend.LOCAL, hooks_uri="http://localhost:9999/hooks", experiment="test_model")

    # get an automatic backend
    backend = MLClient(hooks_uri="http://localhost:9999/hooks")

    backend.add_hook(Hook.RUN_STARTED, 'http://localhost:9999/hooks')

    with backend.start_run() as run:
        model = DummyClassifier()
        X, y = np.array([0, 1, 0, 1, 1, 0]), np.array([1, 0, 1, 0, 0, 1])

        run.log_metric('loss', 0.1)
        run.log_metric('mse', 5)

        run.log_parameter('alpha', 0.05)
        run.log_parameter('stringed', 'str')

        np.save("train_data.npy", X)
        run.log_artifact("train_data.npy", "inputs")

        model.fit(X, y)

        # log a scikit_model
        run.log_model(model, 'model', MLFramework.SCIKIT_LEARN)
        # log a whole directory containing two different models
        run.log_model('resources', 'model', MLFramework.PYFUNC, load_entry_point='my_model.load_stuff')
        # log a single file model
        run.log_model('resources\\test.pkl', 'model', MLFramework.PYFUNC, load_entry_point='my_model.load_stuff')