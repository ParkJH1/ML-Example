import tensorflow as tf
tf1 = tf.compat.v1

g = tf1.Graph()
with g.as_default() as graph:
    node1 = tf1.constant(3.0, tf.float32)
    node2 = tf1.constant(4.0, tf.float32)
    node3 = tf1.add(node1, node2)
    sess = tf1.Session()
    print("node1:", sess.run(node1))
    print("node2:", sess.run(node2))
    print("node3:", sess.run(node3))
