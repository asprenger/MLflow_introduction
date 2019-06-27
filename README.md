# MLflow_introduction

## Setup

Install MLFlow:

    pip install mlflow

## MLFlow Tracking

Start the MLflow tracking server:

    mlflow server --default-artifact-root /tmp/artifacts --backend-store-uri /tmp/mlruns

`--backend-store-uri` defines the location where experiment data and metrics are stored. This can be a local filesystem or a database like SQLite.
`--default-artifact-root` defines the location where artifacts like model files are stored. This can be a local filesystem or a remote filesystem like S3.

By default he tracking server binds to port 5000 on localhost.

Train a set of models with different parameters:

	python train_estimator.py --learning-rate 0.01 --tracking-url http://127.0.0.1:5000
	python train_estimator.py --learning-rate 0.001 --tracking-url http://127.0.0.1:5000
	python train_estimator.py --learning-rate 0.0001 --tracking-url http://127.0.0.1:5000

Each experiment send metrics and training results to the tracking service. Each experiment exports the last checkpoint as TensorFlow SavedModel that gets 
converted to a MLFlow Model and pushed to the tracking service.

Open a browser at `http://127.0.0.1:5000` and select `MNIST_TF_Estimator` to see the 3 runs of the experiment:

![Tracker1](images/tracker1.png?raw=true "Tracker1")

Clicking one of the runs shows the artifacts that have been stored:

![Tracker2](images/tracker1.png?raw=true "Tracker2")

There are the standard files created by the TensorFlow Estimator:

 * Checkpoints
 * GraphDef definition
 * Event files for training and evaluation

There is also the MLFLow Model `exported_model` that has been created by `train_estimator.py`.


## MLFlow Model

A MLflow Model is a standardized packaging format for machine learning models.

Each MLflow Model is a directory containing arbitrary files, together with an MLmodel file in the root of the directory that stores meta-data.


A MLFlow Model can define multiple *flavors* that the model can be used with. A flavour acts as an adapter between the model and a specific framework or tool.
Built-in flavors are:

 * Python function
 * Keras
 * TensorFlow
 * ONNX
 * Spark
 * PyTorch

The script `inference_mlflow_model.py` shows an example how to use a MLflow model in TensorFlow.

First we need to download the MLflow model from the Tracking Service:

    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 mlflow artifacts download --run-id 2eddaed00e264f73b5bd94b057054d7c --artifact-path exported_model

The command copies the model to a local directory and returns the path, e.g.:

    /tmp/artifacts/1/c69616b474964e7fa9f6f6919965d7e5/artifacts/exported_model

The `run-id` is a unique ID and must be looked up in the Tracking Server UI.

Now we can use the model to make predictions:

    python inference_mlflow_model.py file:/tmp/artifacts/1/2eddaed00e264f73b5bd94b057054d7c/artifacts/exported_model


## Using Tensorflow Serving

The default way to deploy a TenssorFlow model is to use TensorFlow Serving. We can actually download the SavedModel from the Tracking Service
and deploy it into Tensorflow Serving. This works because the MLFlow model is just the SavedModel with some meta-data.

We use the same `mlflow artifacts download` command again but this time specify artifact path `exported_model/tfmodel`:

    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 mlflow artifacts download --run-id 2eddaed00e264f73b5bd94b057054d7c --artifact-path exported_model/tfmodel

This downloads the SavedModel to a local directory.

Check out the [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving) documentation how to deploy the model with TensorFlow Serving.


## MLFlow Project

A MLflow Project defines a format to organize and describe code. It also provides an API and command-line tools for running these projects. 
Each MLflow Project has a *MLproject* YAML file that specifies the following properties:

* Name
* Entry points: Commands that can be run within the project, and information about their parameters. 
* Environment: The software environment that should be used to execute project entry points. This includes all library dependencies.

A MLflow Project can be located on the local filesystem or on Github. The environment can be defined as a Conda environment or a Docker container.

For an example check out the MLflow Project in `sample_project`. It defines two endpoints `main` and `validate` and uses a Conda environment.

Run the `main` endpoint:

    touch /tmp/train_dataset.tgz
    mlflow run sample_project -P data_path=/tmp/train_dataset.tgz

Run the `validate` endpoint:

    touch /tmp/test_dataset.tgz
    mlflow run sample_project -e validate -P data_path=/tmp/train_dataset.tgz

At first glance, this might look like an overcomplicated way to run a script. But it is quite useful because you have a single command to execute scripts 
without having to worry about setting up environments and library dependencies.  The `mlflow run` command can reference projects that are hosted on Github. 
This can be combined with a scheduler like Airflow to create recurring workflows.
