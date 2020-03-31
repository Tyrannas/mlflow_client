from mlflow_client import get_auto_backend, LocalBackend, MLFlowBackend, MLFrameworks
from sklearn.dummy import DummyClassifier
import numpy as np

if __name__ == "__main__":
    # backend = get_auto_backend()

    # backend = MLFlowBackend()
    backend = LocalBackend(hooks_uri="http://sgai-api/api/hooks")
    with backend.start_run() as mf:
        model = DummyClassifier()
        X, y = np.array([0, 1, 0, 1, 1, 0]), np.array([1, 0, 1, 0, 0, 1])

        backend.log_metric('loss', 0.1)
        backend.log_metric('mse', 5)

        backend.log_parameter('alpha', 0.05)
        backend.log_parameter('stringed', 'str')

        np.save("train_data.npy", X)
        backend.log_artifact("train_data.npy", "inputs")

        model.fit(X, y)

        # log a scikit_model
        backend.log_model(model, 'model', MLFrameworks.SCIKIT_LEARN)
        # log a whole directory containing two different models
        backend.log_model('resources', 'model', MLFrameworks.PYFUNC, load_entry_point='my_model.load_stuff')
        # log a single file model
        backend.log_model('resources\\test.pkl', 'model', MLFrameworks.PYFUNC, load_entry_point='my_model.load_stuff')