import numpy as np
import mlflow.tensorflow
import tensorflow as tf
import dataset

model_uri = 'file:/tmp/mlruns/1/397b9ae9760a4343b9cc518cbb64448d/artifacts/model'

data_path = '/tmp/mnist'

tf_graph = tf.Graph()
with tf.Session(graph=tf_graph) as sess:
    with tf_graph.as_default():

        ds = dataset.train(data_path)
        next_op = tf.data.make_one_shot_iterator(ds).get_next()


        signature_def = mlflow.tensorflow.load_model(model_uri=model_uri, tf_sess=sess)
        input_tensors = {input_signature.name: tf_graph.get_tensor_by_name(input_signature.name) 
                         for _, input_signature in signature_def.inputs.items()}
        output_tensors = {output_signature.name: tf_graph.get_tensor_by_name(output_signature.name)
                          for _, output_signature in signature_def.outputs.items()}

        x = input_tensors['images:0']
        y_hat = output_tensors['ArgMax:0']

        for _ in range(10):
            image, label = sess.run(next_op)
            pred = sess.run(y_hat, feed_dict={x: np.expand_dims(image, axis=0)})
            print(pred, label)
