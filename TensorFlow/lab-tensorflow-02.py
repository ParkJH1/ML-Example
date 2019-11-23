import tensorflow as tf
tf1 = tf.compat.v1

g = tf1.Graph()
with g.as_default() as graph:
    hello = tf1.constant("Hello TensorFlow!")
    sess = tf1.Session()
    print(sess.run(hello))
