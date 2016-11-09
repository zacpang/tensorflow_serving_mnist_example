import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow.core.framework import types_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('img_binary', '',
                           'I store the mnist pics into npy instead of downloading')

FLAGS = tf.app.flags.FLAGS

host, port = FLAGS.server.split(':')
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

# Generate test data
x = np.load(FLAGS.img_binary)

# Send request
request = predict_pb2.PredictRequest()
request.model_spec.name = 'pzf_mnist_model'
request.inputs['x'].CopyFrom(tf.contrib.util.make_tensor_proto(x, shape=(1, 784)))
request.inputs['keep_prob'].dtype = types_pb2.DT_FLOAT
request.inputs['keep_prob'].float_val.append(1)
result = stub.Predict(request, 10.0)  # 10 secs timeout
print('*********** Result ***********')
print(result)
