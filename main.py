from mlflow_client.core import get_auto_backend, LocalBackend, MLFlowBackend, MLFrameworks
from sklearn.dummy import DummyClassifier
import numpy as np

if __name__ == "__main__":
    backend = get_auto_backend()
    print(backend)

    # backend = MLFlowBackend()
    backend = LocalBackend()
    with backend.start_run():
        model = DummyClassifier()
        X, y = np.array([0, 1, 0, 1, 1, 0]), np.array([1, 0, 1, 0, 0, 1])
        backend.log_metric('loss', 0.1)
        backend.log_metric('mse', 5)
        backend.log_parameter('alpha', 0.05)
        backend.log_parameter('stringed', 'str')
        np.save("experiments/train_data.npy", X)
        backend.log_artifact("train_data.npy", "inputs")
        model.fit(X, y)
        backend.log_model(model, 'dummy/model.pkl', MLFrameworks.SCIKIT_LEARN)
        backend.log_model(model, 'dummy/model2.pkl', MLFrameworks.SCIKIT_LEARN)