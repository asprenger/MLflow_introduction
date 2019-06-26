# MLflow_introduction

## Tracking

Launch the MLflow tracking server:

    mlflow server

The tracking server binds to port 5000 on localhost.

Run a set of experiments:

	python mnist_estimator.py --learning-rate 0.01 --tracking-url http://127.0.0.1:5000
	python mnist_estimator.py --learning-rate 0.001 --tracking-url http://127.0.0.1:5000
	python mnist_estimator.py --learning-rate 0.0001 --tracking-url http://127.0.0.1:5000

Open a browser at `http://127.0.0.1:5000` to check the validation results:




Note that you usually would freeze the graph before creating the SavedModel to reduce model size.


## Project

    touch /tmp/train_dataset.tgz
    mlflow run sample_project -P data_path=/tmp/train_dataset.tgz

    touch /tmp/test_dataset.tgz
    mlflow run sample_project -e validate -P data_path=/tmp/train_dataset.tgz