import json
import os
import sys

import falcon
import numpy as np
import tensorflow as tf

def load_graph(frozen_graph_filename):
    ''' loads serialized tensorflow graph '''

    # load the protobuf file from the disk and parse it to 
    # retrieve the unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


class Model(object):
    def __init__(self, frozen_graph_filename):
        print('Loading the graph')
        graph = load_graph(frozen_graph_filename)
        self.X = graph.get_tensor_by_name('prefix/X:0')
        self.Y = graph.get_tensor_by_name('prefix/Model/Y_predicted:0')
        self.persistent_sess = tf.Session(graph=graph)

    def predict(self, x):
        ''' use model to find prediction '''
        # our graph expect tensors of shape '(?, 1)'
        x = np.array([x]).reshape(-1,1)
        y_hat = self.persistent_sess.run(self.Y, feed_dict={self.X: x})

        return y_hat


class ModelWrapper(object):
    ''' wrapper for tensorflow model '''
    def __init__(self, frozen_graph_filename):
        self.model = Model(frozen_graph_filename)

    def on_get(self, req, resp):
        '''
        processes get request

        $ curl http://0.0.0.0:8000/predict
        '''

        resp.status = falcon.HTTP_200
        resp.body = json.dumps({ 'message' : "Running and waiting" }).encode()
        resp.content_type = 'application/json'

    def on_post(self, req, resp):
        '''
        $ curl -X POST -H "Content-Type: text/pain" --data "10" http://0.0.0.0:8000/predict
        '''
        try:
            raw_data= req.stream.read()
            data = float(raw_data.decode())
            print("Recieved :%s, decoded as: %0.2f" % (raw_data, data))
        except:
            raise falcon.HTTPError(falcon.HTTP_400, 'Could not decode text')

        predicted = self.model.predict(data)
        resp.status = falcon.HTTP_200
        resp.body = json.dumps({ 
            'message' : "Model prediceted %0.2f" % predicted 
        }).encode()
        resp.content_type = 'application/json'


api = application = falcon.API()
api.add_route('/predict', ModelWrapper('./checkpoints/tf/frozen_model.pb'))
